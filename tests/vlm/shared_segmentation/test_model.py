import tempfile
import unittest
import importlib.util
if importlib.util.find_spec("xiaomi_lens") is None:
    raise unittest.SkipTest("Install the pinned shared-segmentation extra to run real-model tests")
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from oa_groundrag.vlm.shared_segmentation.model import tiny_model
from oa_groundrag.vlm.shared_segmentation.contracts import AuxiliaryObservation
from oa_groundrag.vlm.shared_segmentation.session import restore_mask
from oa_groundrag.training.shared_segmentation.engine import configure_trainable,save_delta,load_delta,collate_one,train_bounded
from oa_groundrag.data.shared_segmentation.dataset import JointSample,export_samples,load_samples,assert_parent_isolation
from .helpers import tensor_batch,tiny_session

torch.set_num_threads(2)

def observation(valid=True):
    return AuxiliaryObservation(torch.arange(1024).reshape(1,32,32).float(),torch.full((1,32,32),valid),
                                 'fixture','rgb_grid_verified','encoded')

class SharedModelTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(7)

    def test_official_forward_masks_loss_and_visual_language_gradients(self):
        model=tiny_model();batch=tensor_batch()
        batch['mask_targets']=torch.ones(1,1,32,32)
        out=model(**batch)
        self.assertEqual(out.pred_masks.shape,(1,1,32,32))
        self.assertTrue(torch.isfinite(out.loss))
        out.mask_loss.backward()
        self.assertGreater(float(model.model.visual.patch_embed.proj.weight.grad.norm()),0)
        self.assertGreater(float(model.model.language_model.layers[-1].self_attn.q_proj.weight.grad.norm()),0)

    def test_auxiliary_zero_missing_invalid_and_both_paths(self):
        model=tiny_model().eval();batch=tensor_batch()
        baseline=model(**batch)
        for aux in ({},{'dem':observation(False)},{'dem':observation()}):
            with model.auxiliary_context([aux],[(32,32)]):out=model(**batch)
            torch.testing.assert_close(out.logits,baseline.logits,rtol=0,atol=0)
            torch.testing.assert_close(out.pred_masks,baseline.pred_masks,rtol=0,atol=0)
        torch.nn.init.normal_(model.auxiliary_fusion.adapters['dem'][-1].weight,std=.1)
        with model.auxiliary_context([{'dem':observation()}],[(32,32)]):out=model(**batch)
        self.assertGreater(float((out.logits-baseline.logits).abs().max()),1e-6)
        self.assertGreater(float((out.pred_masks-baseline.pred_masks).abs().max()),1e-6)
        model.zero_grad()
        with model.auxiliary_context([{'dem':observation()}],[(32,32)]):
            model(**batch).logits.square().mean().backward()
        self.assertGreater(float(model.auxiliary_fusion.adapters['dem'][-1].weight.grad.norm()),0)
        invalid=replace(observation(False),values=torch.full((1,32,32),float('nan')))
        with model.auxiliary_context([{'dem':invalid}],[(32,32)]):out=model(**batch)
        torch.testing.assert_close(out.logits,baseline.logits,rtol=0,atol=0)

    def test_multiple_images_explicit_binding_and_padding(self):
        model=tiny_model().eval();batch=tensor_batch(True)
        out=model(**batch)
        self.assertEqual(out.pred_masks.shape,(1,2,32,64))
        self.assertEqual(int(out.pred_mask_valid[0,0,:,32:].sum()),0)
        batch['mask_image_index'][0,0]=2
        with self.assertRaises(ValueError):model(**batch)
        batch['mask_image_index'][0,0]=0
        batch['mask_positions'][0,0]=0
        with self.assertRaises(ValueError):model(**batch)

    def test_invalid_alignment_units_shape(self):
        for item in (replace(observation(),alignment='unknown'),replace(observation(),unit_verified=True),
                     replace(observation(),values=torch.ones(1,16,16))):
            with self.assertRaises(ValueError):item.validate('dem',(32,32))

    def test_mask_restore_crops_padding(self):
        logits=torch.full((64,64),100.);logits[:32,:32]=-100
        mask=restore_mask(logits,(32,32),(19,27))
        self.assertEqual(mask.shape,(19,27));self.assertFalse(mask.any())

    def test_frozen_and_delta_roundtrip(self):
        original=tiny_model();state={n:t.clone() for n,t in original.state_dict().items()}
        configure_trainable(original)
        frozen={n:p.detach().clone() for n,p in original.named_parameters() if not p.requires_grad}
        batch=tensor_batch();batch['mask_targets']=torch.ones(1,1,32,32)
        original(**batch).loss.backward()
        for n,p in original.named_parameters():
            if n in frozen:self.assertIsNone(p.grad)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'delta'
            save_delta(original,root,base_identity={'kind':'random_fixture'},train_config={'rank':8,'visual_last_block':False},report={})
            restored=tiny_model();restored.load_state_dict(state)
            load_delta(restored,root,{'kind':'random_fixture'})
            torch.testing.assert_close(original.eval()(**batch).pred_masks,restored.eval()(**batch).pred_masks)

    def test_real_processor_collation_training_and_generation(self):
        session=tiny_session()
        sample=JointSample('a','a','fixture','train',Image.new('RGB',(32,32),'green'),
                           [np.ones((32,32),dtype=bool)],'Locate candidate.',auxiliaries={'dem':observation()})
        batch=collate_one(session,sample)
        self.assertEqual(int((batch['labels']!=-100).sum()),2)
        with tempfile.TemporaryDirectory() as d:
            report=train_bounded(session,[sample],Path(d)/'delta',base_identity={'kind':'random_fixture'})
            self.assertEqual(report['steps_completed'],2)
        # Real autoregressive generation, no assertion that random text is meaningful.
        _,tokens,_,_=session._generate([sample.image],'Locate candidate.',[sample.auxiliaries])
        self.assertGreater(tokens.numel(),0)

    def test_data_roundtrip_and_parent_isolation(self):
        sample=JointSample('a','parent','fixture','train',Image.new('RGB',(32,32)),[np.zeros((32,32),bool)],'Locate.')
        with self.assertRaises(ValueError):assert_parent_isolation([sample,replace(sample,record_id='b',split='val')])
        with self.assertRaises(ValueError):replace(sample,description={},mask_origin='prediction').validate()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'data';export_samples([sample],root)
            self.assertEqual(load_samples(root)[0].parent_id,'parent')
            with self.assertRaises(Exception):export_samples([sample],root)

    def test_production_loader_roundtrip_and_identity_rejection(self):
        import json
        from oa_groundrag.artifacts.identity import sha256_file
        from oa_groundrag.vlm.shared_segmentation.contracts import LENS_REVISION
        from oa_groundrag.vlm.shared_segmentation.session import SharedModelSession
        session=tiny_session()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'model';root.mkdir()
            session.model.save_pretrained(root)
            session.processor.save_pretrained(root)
            ledger=Path(d)/'identity.json'
            ledger.write_text(json.dumps({'kind':'geo_groundrag','lens_revision':LENS_REVISION,
                                          'files':{p.name:sha256_file(p) for p in root.iterdir() if p.is_file()}}))
            restored=SharedModelSession.load(root,ledger,max_pixels=1024)
            batch=tensor_batch()
            torch.testing.assert_close(session.model(**batch).pred_masks,restored.model(**batch).pred_masks)
            (root/'extra.txt').write_text('unexpected')
            with self.assertRaises(ValueError):SharedModelSession.load(root,ledger)

    def test_joint_training_loop_accepts_explicit_cpu_budget(self):
        from oa_groundrag.training.shared_segmentation.engine import train_joint
        session=tiny_session()
        sample=JointSample('loop','parent','fixture','train',Image.new('RGB',(32,32)),[np.ones((32,32),bool)],'Locate.')
        with tempfile.TemporaryDirectory() as d:
            report=train_joint(session,[sample],Path(d)/'loop',base_identity={'kind':'random_fixture'},steps=3)
            self.assertEqual(report['steps_completed'],3)
            with self.assertRaises(ValueError):train_bounded(session,[sample],Path(d)/'bad',base_identity={},steps=3)
