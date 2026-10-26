import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from oa_groundrag.runtime.shared_model import GeoGroundRuntime, save_result
from oa_groundrag.vlm.shared_segmentation.session import SegmentationResult
from oa_groundrag.grounding.outputs import region_output_template


def observation():
    return region_output_template('target_present')

class Session:
    def __init__(self,masks=None,status='target_present'):
        self.masks=masks if masks is not None else [np.eye(32,dtype=bool)]
        self.status=status
        self.feedback=[]
        self.forged=False
    def segment(self,*args,**kwargs):
        return SegmentationResult(self.status,self.masks,[observation() for _ in self.masks],'test fixture')
    def describe_region(self,image,mask,aux):
        self.feedback.append(mask.copy());return observation()
    def generate_text(self,messages):
        return json.dumps(dict(schema_version='oa_groundrag.text_rag.pass2_output.v1',
            supporting_interpretations=[dict(text='该候选区域仍需结合独立证据核验。',evidence_ids=['fake' if self.forged else 'e1'])],
            alternative_explanations=[],limitations=[],recommended_verification=[],summary=dict(text='目前证据有限。',evidence_ids=['e1'])))

class Retriever:
    def __init__(self,mutate=False):self.mutate=mutate
    def retrieve(self,**kwargs):
        if self.mutate:kwargs['program_facts']['changed']=True
        return SimpleNamespace(packet={'items':[dict(evidence_id='e1',knowledge_type='interpretation',
            source_id='source',source_title='fixture',pdf_page=1,section='fixture',content='证据不足时需要进一步核验。',conditions=[])]},
            bank_identity={'kind':'fixture'})

class RuntimeTests(unittest.TestCase):
    def test_feedback_uses_actual_masks_and_geometry(self):
        session=Session([np.eye(32,dtype=bool),np.fliplr(np.eye(32,dtype=bool))])
        out=GeoGroundRuntime(session).run(Image.new('RGB',(32,32)),'Describe.')
        self.assertEqual(len(out['regions']),2)
        self.assertEqual(out['regions'][0]['program_facts']['mask']['area_pixels'],32)
        np.testing.assert_array_equal(session.feedback[1],session.masks[1])
        with tempfile.TemporaryDirectory() as d:
            public=save_result(out,Path(d)/'result')
            self.assertTrue((Path(d)/'result'/public['regions'][0]['mask_path']).exists())

    def test_joint_does_not_feedback(self):
        session=Session();GeoGroundRuntime(session).run(Image.new('RGB',(32,32)),'Describe.',mode='joint')
        self.assertEqual(session.feedback,[])

    def test_no_target_and_incomplete_are_distinct(self):
        for status in ('no_target','incomplete','invalid_output'):
            result=GeoGroundRuntime(Session([],status)).run(Image.new('RGB',(32,32)),'Describe.')
            self.assertEqual(result['status'],status);self.assertEqual(result['regions'],[])

    def test_rag_cannot_mutate_facts(self):
        with self.assertRaises(RuntimeError):GeoGroundRuntime(Session(),Retriever(True)).run(Image.new('RGB',(32,32)),'Why?')

    def test_rag_sources_and_fake_citation_rejected(self):
        session=Session();runtime=GeoGroundRuntime(session,Retriever())
        output=runtime.run(Image.new('RGB',(32,32)),'Why?')
        self.assertEqual(output['regions'][0]['interpretation']['sources'][0]['evidence_id'],'e1')
        session.forged=True
        with self.assertRaises(Exception):runtime.run(Image.new('RGB',(32,32)),'Why?')

    def test_knowledge_qa_does_not_segment(self):
        session=Session()
        session.segment=lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not segment'))
        GeoGroundRuntime(session,Retriever()).knowledge_qa('What is a landslide?')
