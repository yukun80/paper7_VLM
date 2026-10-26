"""One Lens model instance for segmentation, region feedback, and text explanation."""
from dataclasses import dataclass
from pathlib import Path
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from oa_groundrag.artifacts.identity import sha256_file
from oa_groundrag.grounding.outputs import parse_region_model_output, region_output_contract
from .contracts import LENS_REVISION, SCHEMA, validate_auxiliaries

@dataclass
class SegmentationResult:
    status: str
    masks: list[np.ndarray]
    observations: list[dict]
    raw_text: str
    reason: str | None = None


def restore_mask(logits, canvas_hw, original_hw):
    h,w=canvas_hw
    if h>logits.shape[-2] or w>logits.shape[-1] or min(h,w,*original_hw)<=0:
        raise ValueError('Invalid mask geometry')
    cropped=logits[:h,:w].float()[None,None]
    restored=F.interpolate(cropped,size=original_hw,mode='bilinear',align_corners=False)[0,0]
    return (restored>0).cpu().numpy()


def checkpoint_identity(root: Path, *, expected_manifest: Path) -> dict:
    root=Path(root)
    from oa_groundrag.artifacts.io import first_symlink_component
    if first_symlink_component(root) is not None:
        raise ValueError("Model root contains a symbolic link")
    ledger=json.loads(Path(expected_manifest).read_text())
    if ledger.get('lens_revision')!=LENS_REVISION or ledger.get('kind') not in {'xiaomi_lens','geo_groundrag'}:
        raise ValueError('Unverified model family/source revision')
    files=ledger['files']
    if not {'config.json','tokenizer_config.json'}.issubset(files) or not any(n.endswith('.safetensors') for n in files):
        raise ValueError('Incomplete model identity ledger')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=set(files):
        raise ValueError('Model file inventory differs from its ledger')
    for name,digest in files.items():
        p=root/name
        if '..' in Path(name).parts or Path(name).is_absolute() or p.is_symlink() or sha256_file(p)!=digest:
            raise ValueError('Model file identity mismatch')
    return ledger


class SharedModelSession:
    def __init__(self, model, processor, *, max_new_tokens=512, max_pixels=262144):
        self.model=model.eval()
        self.processor=processor
        self.max_new_tokens=max_new_tokens
        if max_pixels < 1024 or max_new_tokens < 1:
            raise ValueError('Invalid image or generation budget')
        self.processor.image_processor.min_pixels=1024
        self.processor.image_processor.max_pixels=max_pixels
        self.processor.image_processor.size={'shortest_edge':1024,'longest_edge':max_pixels}
        self.max_pixels=max_pixels
        self.mask_token_id=processor.tokenizer.convert_tokens_to_ids('<|mask|>')
        if processor.tokenizer.encode('<|mask|>',add_special_tokens=False)!=[self.mask_token_id] or self.mask_token_id==processor.tokenizer.unk_token_id:
            raise ValueError('Tokenizer has no single registered mask token')
        if getattr(model.config,'mask_token_id',self.mask_token_id)!=self.mask_token_id:
            raise ValueError('Model/tokenizer mask token identity mismatch')
        if not 0 <= self.mask_token_id < model.get_input_embeddings().weight.shape[0]:
            raise ValueError('Mask token has no model embedding')
        model.config.mask_token_id=self.mask_token_id

    @classmethod
    def load(cls, root, identity_manifest, *, device='cpu', **kwargs):
        from transformers import AutoProcessor
        from .model import GeoGroundModel
        identity=checkpoint_identity(Path(root),expected_manifest=Path(identity_manifest))
        model,info=GeoGroundModel.from_pretrained(str(root),local_files_only=True,attn_implementation='sdpa',
            dtype=torch.float32 if device=='cpu' else torch.bfloat16,output_loading_info=True,device_map={'':device})
        missing=info.get('missing_keys',[])
        if info.get('unexpected_keys') or info.get('mismatched_keys') or any(not k.startswith('auxiliary_fusion.') for k in missing):
            raise ValueError('Checkpoint architecture does not match Lens')
        if identity['kind']=='geo_groundrag' and missing:
            raise ValueError('Geo checkpoint missing trained tensors')
        if missing:
            for adapter in model.auxiliary_fusion.adapters.values():
                torch.nn.init.zeros_(adapter[-1].weight);torch.nn.init.zeros_(adapter[-1].bias)
        processor=AutoProcessor.from_pretrained(str(root),local_files_only=True,trust_remote_code=False)
        session=cls(model.to(device),processor,**kwargs)
        session.identity=identity
        return session

    def inputs(self, images, prompt, *, answer=None):
        content=[{'type':'image','image':im} for im in images]+[{'type':'text','text':prompt}]
        messages=[{'role':'user','content':content}]
        if answer is not None:
            messages.append({'role':'assistant','content':[{'type':'text','text':answer}]})
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=answer is None,enable_thinking=False)
        inputs=self.processor(text=[text],images=images or None,return_tensors='pt',padding=True,add_special_tokens=False)
        grid=inputs.get('image_grid_thw')
        if grid is not None and bool((grid[:,1]*grid[:,2]*256 > self.max_pixels).any()):
            raise ValueError('Processor exceeded the configured image pixel budget')
        device=next(self.model.parameters()).device
        return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}

    def _generate(self, images, prompt, auxiliaries):
        inputs=self.inputs(images,prompt)
        sizes=[(im.height,im.width) for im in images]
        with self.model.auxiliary_context(auxiliaries,sizes),torch.inference_mode():
            output=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False)
        completion=output[:,inputs['input_ids'].shape[1]:]
        eos=self.model.generation_config.eos_token_id
        eos={eos} if isinstance(eos,int) else set(eos or [])
        complete=bool(completion.numel() and int(completion[0,-1]) in eos)
        raw=self.processor.tokenizer.decode(completion[0],skip_special_tokens=False)
        # Preserve mask tokens but strip only terminal chat/control tokens and empty thinking.
        raw=re.sub(r'<think>.*?</think>','',raw,flags=re.S).strip()
        for token in ('<|im_end|>','<|endoftext|>'):
            raw=raw.replace(token,'')
        return inputs,completion,raw.strip(),complete

    def segment(self,image,instruction,auxiliaries=None, *, joint_description=True):
        aux=auxiliaries or {};validate_auxiliaries(aux,(image.height,image.width))
        contract=region_output_contract('target_present')
        prompt=(instruction+'\nReturn exactly NO_TARGET when no candidate is visible. Otherwise emit one <|mask|> per region, then JSON '
                '{"regions": [one observation per mask in the same order]}. Observation contract: '+json.dumps(contract,ensure_ascii=False)+
                '\nDescribe candidates, never confirmed landslides. Do not guess numerical geometry or physical auxiliary units.')
        if not joint_description:
            prompt=instruction+'\nLocate candidate landslide regions and report bbox coordinates and masks. Return exactly NO_TARGET if absent. Do not confirm landslides.'
        inputs,completion,raw,complete=self._generate([image],prompt,[aux])
        if not complete:
            return SegmentationResult('incomplete',[],[],raw,'generation_length_limit')
        count=int((completion==self.mask_token_id).sum())
        if raw=='NO_TARGET' and count==0:
            return SegmentationResult('no_target',[],[],raw)
        if count==0:
            return SegmentationResult('invalid_output',[],[],raw,'missing_mask_tokens')
        explicit_empty = raw.replace('<|mask|>', '').strip() == 'NO_TARGET'
        observations = []
        if not explicit_empty and joint_description:
            try:
                start=raw.index('{'); payload=json.loads(raw[start:])
                if set(payload)!={'regions'} or len(payload['regions'])!=count:
                    raise ValueError('Mask/description count mismatch')
                observations=[parse_region_model_output(v).to_dict() for v in payload['regions']]
                if any(o['target_status']!='target_present' for o in observations):
                    raise ValueError('Contradictory target state')
            except Exception as error:
                return SegmentationResult('invalid_output',[],[],raw,str(error))
        full=torch.cat((inputs['input_ids'],completion),1)
        replay={**inputs,'input_ids':full,'attention_mask':torch.ones_like(full),'use_cache':False,'logits_to_keep':1}
        if 'mm_token_type_ids' in replay:
            replay['mm_token_type_ids']=torch.cat((replay['mm_token_type_ids'],torch.zeros_like(completion)),1)
        replay['mask_positions']=(completion==self.mask_token_id).nonzero()[:,1][None]+inputs['input_ids'].shape[1]
        replay['mask_image_index']=torch.zeros_like(replay['mask_positions'])
        with self.model.auxiliary_context([aux],[(image.height,image.width)]),torch.inference_mode():
            decoded=self.model(**replay)
        grid=inputs['image_grid_thw'][0]
        masks=[restore_mask(x,(int(grid[1])*16,int(grid[2])*16),(image.height,image.width)) for x in decoded.pred_masks[0]]
        if explicit_empty:
            if any(m.any() for m in masks):
                return SegmentationResult('invalid_output',masks,[],raw,'NO_TARGET contradicts nonempty masks')
            return SegmentationResult('no_target',[],[],raw)
        if any(not m.any() for m in masks):
            return SegmentationResult('invalid_output',masks,observations,raw,'empty_mask_with_target_description')
        return SegmentationResult('target_present',masks,observations,raw)

    def region_feedback_input(self,image,mask,auxiliaries=None):
        from oa_groundrag.grounding.evidence import deterministic_mask_facts
        if mask.dtype!=np.bool_ or mask.shape!=(image.height,image.width) or not mask.any():
            raise ValueError('Region feedback requires a nonempty aligned predicted mask')
        facts=deterministic_mask_facts(mask)
        x0,y0,x1,y1=facts['bbox_xyxy_pixel_half_open']
        crop=image.crop((max(0,x0-8),max(0,y0-8),min(image.width,x1+8),min(image.height,y1+8)))
        images=[image,Image.fromarray(mask.astype('uint8')*255).convert('RGB'),crop]
        prompt=('Image 1: original optical image. Image 2: binary region POSITION ONLY, white is not a land-cover colour. '
                'Image 3: optical context crop, its edge is not the target boundary. Describe only the specified region. '
                'Do not generate masks or numerical geometry. Return strict JSON matching '+json.dumps(region_output_contract('target_present'),ensure_ascii=False)+
                '\nRead-only computed facts: '+json.dumps(facts)+'\nAuxiliary restrictions: '+json.dumps([o.evidence(n) for n,o in (auxiliaries or {}).items()]))
        return images,prompt

    def describe_region(self,image,mask,auxiliaries=None):
        images,prompt=self.region_feedback_input(image,mask,auxiliaries)
        _,_,raw,complete=self._generate(images,prompt,[auxiliaries or {},{},{}])
        if not complete:
            raise ValueError('Incomplete region description')
        observation=parse_region_model_output(raw).to_dict()
        if observation['target_status']!='target_present':
            raise ValueError('Region description contradicts the actual mask')
        return observation

    def generate_text(self,messages):
        prompt=self.processor.apply_chat_template(list(messages),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=self.processor(text=[prompt],return_tensors='pt',add_special_tokens=False)
        device=next(self.model.parameters()).device
        inputs={k:v.to(device) for k,v in inputs.items()}
        with torch.inference_mode():
            out=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False)
        return self.processor.tokenizer.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
