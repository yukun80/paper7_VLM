import json
import unittest
import importlib.util
if importlib.util.find_spec("xiaomi_lens") is None:
    raise unittest.SkipTest("Install the pinned shared-segmentation extra to run real-model tests")
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from oa_groundrag.grounding.outputs import region_output_template
from oa_groundrag.vlm.shared_segmentation.session import SharedModelSession
from oa_groundrag.evaluation.shared_segmentation import segmentation_metrics,compare_paired
from .helpers import tiny_session,tensor_batch

class SessionContractTests(unittest.TestCase):
    def test_absence_truncation_and_missing_mask_are_distinct(self):
        session=tiny_session();image=Image.new('RGB',(32,32))
        for text,done,expected in [('NO_TARGET',True,'no_target'),('NO_TARGET',False,'incomplete'),('nonsense',True,'invalid_output')]:
            session._generate=lambda *a,**k: ({},torch.tensor([[2]]),text,done)
            self.assertEqual(session.segment(image,'Locate.').status,expected)

    def test_feedback_changes_position_image_for_wrong_mask(self):
        session=tiny_session();seen=[]
        def capture(images,prompt,aux):
            seen.append((np.asarray(images[1]).copy(),aux))
            return {},torch.tensor([[2]]),json.dumps(region_output_template('target_present')),True
        session._generate=capture
        first=np.zeros((32,32),bool);first[:5,:5]=True
        second=np.zeros((32,32),bool);second[20:,20:]=True
        image=Image.new('RGB',(32,32))
        session.describe_region(image,first);session.describe_region(image,second)
        self.assertFalse(np.array_equal(seen[0][0],seen[1][0]))
        self.assertEqual(seen[0][1],[{},{},{}])
        with self.assertRaises(ValueError):session.describe_region(image,np.zeros((32,32),bool))

    def test_mask_targets_and_padding_validation(self):
        session=tiny_session();batch=tensor_batch()
        batch['mask_targets']=torch.ones(1,2,32,32)
        with self.assertRaises(ValueError):session.model(**batch)
        batch['mask_targets']=torch.ones(1,1,32,32)*float('nan')
        with self.assertRaises(ValueError):session.model(**batch)
        batch['mask_targets']=torch.ones(1,1,32,32)
        batch['mask_valid_mask']=torch.zeros(1,1)
        batch['mask_pixel_valid']=torch.ones_like(batch['mask_targets'])
        with self.assertRaises(ValueError):session.model(**batch)
        batch['mask_pixel_valid'].zero_()
        self.assertTrue(torch.isfinite(session.model(**batch).loss))

    def test_metrics_empty_false_positive_and_wrong_mask(self):
        empty=np.zeros((8,8),bool);target=empty.copy();target[:2,:2]=True
        wrong=empty.copy();wrong[6:,6:]=True
        self.assertEqual(segmentation_metrics(empty,empty)['iou'],1)
        self.assertFalse(segmentation_metrics(target,empty)['no_target_correct'])
        self.assertEqual(segmentation_metrics(wrong,target)['iou'],0)
        a=[{'record_id':'a','truth_sha256':'x','split':'val','metrics':{'iou':.2}}]
        b=[{'record_id':'a','truth_sha256':'x','split':'val','metrics':{'iou':.4}}]
        self.assertAlmostEqual(compare_paired(a,b)['iou_delta'],.2)
        b[0]['truth_sha256']='wrong'
        with self.assertRaises(ValueError):compare_paired(a,b)

    def test_processor_budget_applies_to_real_sized_images(self):
        session=tiny_session()
        inputs=session.inputs([Image.new('RGB',(224,224))],'Locate.')
        self.assertEqual(inputs['image_grid_thw'].tolist(),[[1,2,2]])

    def test_scoring_binds_actual_input_and_rejects_train(self):
        import tempfile
        from dataclasses import replace
        from oa_groundrag.runtime.shared_model import GeoGroundRuntime,save_result
        from oa_groundrag.data.shared_segmentation.dataset import JointSample
        from oa_groundrag.evaluation.shared_segmentation import score_artifact
        from .test_runtime import Session
        image=Image.new('RGB',(32,32));mask=np.eye(32,dtype=bool)
        sample=JointSample('sample','parent','fixture','val',image,[mask],'Locate.')
        result=GeoGroundRuntime(Session([mask])).run(image,'Locate.',record_id='sample')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'result';save_result(result,root)
            self.assertEqual(score_artifact(root,sample)['metrics']['iou'],1)
            with self.assertRaises(ValueError):score_artifact(root,replace(sample,split='train'))
            with self.assertRaises(ValueError):score_artifact(root,replace(sample,record_id='wrong'))
