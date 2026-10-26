"""Geo-GroundRAG orchestration: one shared VLM; existing read-only Text RAG."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from oa_groundrag.grounding.evidence import deterministic_mask_facts
from oa_groundrag.retrieval.contracts import TextRagTask, RagMode
from oa_groundrag.retrieval.pass2 import build_pass2_messages, parse_pass2_output
from oa_groundrag.vlm.shared_segmentation.contracts import SCHEMA

class ExistingTextRAG:
    def __init__(self,config_path: Path, *, device='cpu'):
        from oa_groundrag.retrieval.runtime_config import load_text_rag_runtime_config
        from oa_groundrag.retrieval.runtime import RuntimeTextRetriever
        config=load_text_rag_runtime_config(config_path)
        config=replace(config,dense=replace(config.dense,device=device))
        self.retriever=RuntimeTextRetriever(config)

    def retrieve(self,**kwargs):
        return self.retriever.retrieve(**kwargs)

    def release(self):self.retriever.release()


def immutable_signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def region_auxiliary_evidence(name,observation,mask):
    evidence=observation.evidence(name)
    region=torch.from_numpy(mask.copy()).to(observation.valid.device)[None]
    valid=observation.valid & region
    evidence['scope']='predicted_region'
    evidence['valid_region_pixels']=int(valid.sum())
    evidence['region_coverage']=float(valid.sum())/max(1,int(region.sum()))
    evidence['physical_interpretation_allowed'] &= bool(valid.any())
    if evidence['physical_interpretation_allowed']:
        values=observation.values[valid]
        evidence['measured_summary']={'minimum':float(values.min()),'mean':float(values.mean()),'maximum':float(values.max()),'unit':observation.unit}
    return evidence


class GeoGroundRuntime:
    def __init__(self,session,retriever=None):
        self.session=session
        self.retriever=retriever

    def _interpret(self,question,record_id,observation,facts,status,modalities,task):
        if self.retriever is None:return {'status':'disabled','sources':[]}
        before=immutable_signature([observation,facts])
        result=self.retriever.retrieve(task=task,record_id=record_id,question=question,
            observation=observation,program_facts=facts,target_status=status,
            candidate_count=int(status=='target_present'),available_modalities=modalities)
        if immutable_signature([observation,facts])!=before:raise RuntimeError('RAG modified upstream evidence')
        packet=deepcopy(result.packet)
        packet_before=immutable_signature(packet)
        messages=build_pass2_messages(question=question,target_status=status,program_facts=facts,
                                      observation=observation,packet=packet,task=task)
        raw=self.session.generate_text(messages)
        if immutable_signature([observation,facts])!=before or immutable_signature(packet)!=packet_before:
            raise RuntimeError('Generation modified upstream evidence')
        parsed=parse_pass2_output(raw,mode=RagMode.TEXT_RAG,packet=packet).to_dict()
        sources=[{k:item[k] for k in ('evidence_id','source_id','source_title','pdf_page','section')} for item in packet['items']]
        return {'status':'generated','output':parsed,'sources':sources,'bank_identity':dict(result.bank_identity)}

    def run(self,image,question, *, auxiliaries=None,mode='feedback',record_id='user-image'):
        if mode not in {'joint','feedback'}:raise ValueError('Unknown description mode')
        aux=auxiliaries or {}
        available=('optical',*[name for name,obs in aux.items() if bool(obs.valid.any())])
        predicted=self.session.segment(image,question,aux,joint_description=mode=='joint')
        output=dict(schema_version=SCHEMA,record_id=record_id,image_sha256=hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest(),status=predicted.status,description_mode=mode,
                    prediction_reason=predicted.reason,raw_generation=predicted.raw_text,regions=[],scientific_acceptance=False,
                    model_identity=getattr(self.session,'identity',{'kind':'unverified_or_fixture'}),
                    auxiliary_evidence=[o.evidence(n) for n,o in aux.items()])
        if predicted.status not in {'target_present','no_target'}:
            return output
        if predicted.status=='no_target':
            facts={'mask':{'area_pixels':0},'candidate_count':0,'auxiliary_observations':output['auxiliary_evidence']}
            output['interpretation']=self._interpret(question,record_id,{},facts,'no_target',available,TextRagTask.CANDIDATE_INTERPRETATION)
            return output
        if mode=='joint' and len(predicted.masks)!=len(predicted.observations):raise ValueError('Region count mismatch')
        for i,mask in enumerate(predicted.masks):
            before=hashlib.sha256(mask.tobytes()).hexdigest()
            observation=predicted.observations[i] if mode=='joint' else self.session.describe_region(image,mask,aux)
            facts={'mask':deterministic_mask_facts(mask),'image':{'height':image.height,'width':image.width}}
            facts['auxiliary_observations']=[region_auxiliary_evidence(name,obs,mask) for name,obs in aux.items()]
            interpretation=self._interpret(question,f'{record_id}:region-{i}',observation,facts,'target_present',
                                            available,TextRagTask.CANDIDATE_INTERPRETATION)
            if hashlib.sha256(mask.tobytes()).hexdigest()!=before:raise RuntimeError('Upstream mask was modified')
            output['regions'].append(dict(region_id=f'region-{i}',mask=mask,mask_sha256=before,
                mask_origin='prediction',observation=observation,program_facts=facts,interpretation=interpretation))
        return output

    def knowledge_qa(self,question):
        return self._interpret(question,'knowledge-qa',{}, {},'no_target',('general',),TextRagTask.PROFESSIONAL_QA)


def save_result(result,root: Path):
    from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
    from PIL import Image
    with AtomicArtifactDirectory(root) as writer:
        public=deepcopy(result)
        for i,region in enumerate(public['regions']):
            mask=region.pop('mask')
            name=f'region-{i}.png'
            Image.fromarray(mask.astype(np.uint8)*255).save(writer.path(name))
            region['mask_path']=name
        writer.write_json('result.json',public)
        from oa_groundrag.artifacts.identity import sha256_file
        writer.write_json('manifest.json',{'schema_version':SCHEMA,'files':{p.name:sha256_file(p) for p in writer.staging.iterdir() if p.is_file()},'scientific_acceptance':False})
        writer.publish()
    return public
