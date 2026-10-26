"""Bounded joint training, explicit supervision, and portable trainable deltas."""
from pathlib import Path
import json
import time
import torch
import torch.nn.functional as F
from oa_groundrag.artifacts.directory import AtomicArtifactDirectory
from oa_groundrag.artifacts.identity import sha256_file
from oa_groundrag.vlm.shared_segmentation.contracts import SCHEMA, LENS_REVISION


def configure_trainable(model, *, rank=8, visual_last_block=False):
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad_(False)
    targets=[name for name,module in model.model.language_model.named_modules()
             if name.endswith(('q_proj','k_proj','v_proj','o_proj')) and isinstance(module,torch.nn.Linear)]
    if not targets:
        raise ValueError('No verified language attention projections')
    model.model.language_model=get_peft_model(model.model.language_model,
        LoraConfig(r=rank,lora_alpha=2*rank,lora_dropout=0.0,target_modules=targets,bias='none'))
    for module in (model.mask_decoder,model.auxiliary_fusion):
        for p in module.parameters():p.requires_grad_(True)
    if visual_last_block:
        for p in model.model.visual.blocks[-1].parameters():p.requires_grad_(True)
    return {name:p.numel() for name,p in model.named_parameters() if p.requires_grad}


def collate_one(session,sample,max_tokens=2048):
    """Batch size one; no silent truncation, mask targets follow processor canvas."""
    sample.validate()
    if hasattr(sample,"messages"):
        return collate_replay(session,sample,max_tokens)
    if sample.mask_origin!='ground_truth':
        raise ValueError('Training requires identified GT supervision')
    answer='<|mask|>'*len(sample.masks)
    if sample.masks and not any(m.any() for m in sample.masks):
        answer+='NO_TARGET'
    elif sample.description is not None:
        if len(sample.masks)!=1:
            raise ValueError('One compact description cannot supervise multiple regions')
        answer+='\n'+json.dumps({'regions':[sample.description]},ensure_ascii=False)
    elif not sample.masks:
        raise ValueError('Text-only rows require a separate explicit text target')
    prompt=sample.instruction
    # A selected GT subregion is a referring task, not an unconditional whole-image label.
    if sample.description is not None:
        from oa_groundrag.grounding.evidence import deterministic_mask_facts
        if sample.masks[0].any():
            prompt+='\nDescribe the candidate selected by this pixel bbox: '+str(deterministic_mask_facts(sample.masks[0])['bbox_xyxy_pixel_half_open'])
    inputs=session.inputs([sample.image],prompt,answer=answer)
    prefix=session.inputs([sample.image],prompt)
    n=prefix['input_ids'].shape[1]
    if inputs['input_ids'].shape[1]>max_tokens:
        raise ValueError('Token budget exceeded; sample was not truncated')
    if not torch.equal(inputs['input_ids'][:,:n],prefix['input_ids']):
        raise ValueError('Chat template does not preserve assistant supervision prefix')
    labels=inputs['input_ids'].clone();labels[:,:n]=-100
    positions=(inputs['input_ids'][0,n:]==session.mask_token_id).nonzero().flatten()+n
    if len(positions)!=len(sample.masks):
        raise ValueError('Mask supervision/token mismatch')
    h,w=map(int,inputs['image_grid_thw'][0,1:].tolist())
    device=inputs['input_ids'].device
    targets=torch.stack([torch.from_numpy(m.copy()) for m in sample.masks]).float().to(device)
    targets=F.interpolate(targets[:,None],size=(h*16,w*16),mode='nearest')[:,0][None]
    return {**inputs,'labels':labels,'mask_positions':positions[None],
            'mask_image_index':torch.zeros_like(positions)[None],'mask_targets':targets,
            'mask_pixel_valid':torch.ones_like(targets),'mask_valid_mask':torch.ones_like(positions)[None],
            'mask_canvas_grid':(h,w),'use_cache':False}


def collate_region_description(session,sample,max_tokens=2048):
    """GT-mask conditioned text supervision, explicitly separate from prediction evaluation."""
    sample.validate()
    if sample.description is None or len(sample.masks)!=1 or not sample.masks[0].any():
        raise ValueError('Feedback supervision requires a nonempty identified GT region and description')
    images,prompt=session.region_feedback_input(sample.image,sample.masks[0],sample.auxiliaries)
    answer=json.dumps(sample.description,ensure_ascii=False)
    batch=session.inputs(images,prompt,answer=answer)
    prefix=session.inputs(images,prompt)
    n=prefix['input_ids'].shape[1]
    if batch['input_ids'].shape[1]>max_tokens or not torch.equal(batch['input_ids'][:,:n],prefix['input_ids']):
        raise ValueError('Feedback budget or supervision prefix mismatch')
    labels=batch['input_ids'].clone();labels[:,:n]=-100
    return {**batch,'labels':labels,'use_cache':False},[sample.auxiliaries,{},{}],[(im.height,im.width) for im in images]


def collate_replay(session,sample,max_tokens=2048):
    processor=session.processor
    text=processor.apply_chat_template(sample.messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
    prefix=processor.apply_chat_template(sample.messages[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=processor(text=[text],images=sample.images or None,return_tensors='pt',add_special_tokens=False)
    prompt=processor(text=[prefix],images=sample.images or None,return_tensors='pt',add_special_tokens=False)
    n=prompt['input_ids'].shape[1]
    if inputs['input_ids'].shape[1]>max_tokens or not torch.equal(inputs['input_ids'][:,:n],prompt['input_ids']):
        raise ValueError('Replay text budget/prefix mismatch')
    device=next(session.model.parameters()).device
    inputs={k:v.to(device) for k,v in inputs.items()}
    labels=inputs['input_ids'].clone();labels[:,:n]=-100
    return {**inputs,'labels':labels,'use_cache':False}


def save_delta(model, output: Path, *, base_identity: dict, train_config: dict, report: dict):
    from safetensors.torch import save_file
    with AtomicArtifactDirectory(output) as writer:
        tensors={n:p.detach().cpu().contiguous() for n,p in model.named_parameters() if p.requires_grad}
        save_file(tensors,str(writer.path('trainable.safetensors')))
        writer.write_json('manifest.json',dict(schema_version=SCHEMA,lens_revision=LENS_REVISION,
            base_identity=base_identity,train_config=train_config,tensors={n:list(t.shape) for n,t in tensors.items()},
            weights_sha256=sha256_file(writer.path('trainable.safetensors')),scientific_acceptance=False))
        writer.write_json('report.json',report)
        writer.publish()


def load_delta(model, root: Path, base_identity: dict):
    from safetensors.torch import load_file
    meta=json.loads((root/'manifest.json').read_text())
    if meta['schema_version']!=SCHEMA or meta['base_identity']!=base_identity or meta['lens_revision']!=LENS_REVISION:
        raise ValueError('Delta/base identity mismatch')
    if sha256_file(root/'trainable.safetensors')!=meta['weights_sha256']:
        raise ValueError('Delta weights identity mismatch')
    configure_trainable(model,**meta['train_config'])
    tensors=load_file(str(root/'trainable.safetensors'))
    params=dict(model.named_parameters())
    if set(tensors)!={n for n,p in params.items() if p.requires_grad} or {n:list(t.shape) for n,t in tensors.items()}!=meta['tensors']:
        raise ValueError('Delta trainable parameter set mismatch')
    with torch.no_grad():
        for name,t in tensors.items():params[name].copy_(t.to(params[name]))


def train_joint(session,samples,output: Path, *, base_identity: dict,steps=2,timeout_seconds=600,rank=8,visual_last_block=False,description_supervision="joint",learning_rate=1e-4):
    from oa_groundrag.data.shared_segmentation.dataset import assert_parent_isolation
    if Path(output).exists():raise FileExistsError(output)
    if description_supervision not in {'joint','feedback'}:raise ValueError('Unknown description supervision')
    if steps<1 or len(samples)<1 or timeout_seconds<=0 or learning_rate<=0:
        raise ValueError('Invalid training budget or learning rate')
    if any(s.split!='train' for s in samples):raise ValueError('Training cannot consume val')
    assert_parent_isolation(samples)
    model=session.model
    config=dict(rank=rank,visual_last_block=visual_last_block)
    trainable=configure_trainable(model,**config)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=learning_rate,weight_decay=0.0)
    model.train();start=time.monotonic();losses=[]
    cuda=next(model.parameters()).is_cuda
    if cuda:torch.cuda.reset_peak_memory_stats()
    for i in range(steps):
        if time.monotonic()-start>timeout_seconds:raise TimeoutError('Bounded training time budget')
        sample=samples[i%len(samples)]
        if description_supervision=='feedback' and getattr(sample,'description',None) is not None and sample.masks[0].any():
            batch,auxiliary,spatial_sizes=collate_region_description(session,sample)
        else:
            batch=collate_one(session,sample)
            auxiliary=[{} for _ in sample.images] if hasattr(sample,'messages') else [sample.auxiliaries]
            spatial_sizes=[(im.height,im.width) for im in sample.images] if hasattr(sample,'messages') else [(sample.image.height,sample.image.width)]
        optimizer.zero_grad(set_to_none=True)
        with model.auxiliary_context(auxiliary,spatial_sizes):
            out=model(**batch)
            if not torch.isfinite(out.loss):raise FloatingPointError('Non-finite joint loss')
            out.loss.backward()
        for name,p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError(name)
            if not p.requires_grad and p.grad is not None:raise RuntimeError('Frozen parameter received gradient')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        optimizer.step()
        losses.append(dict(total=float(out.loss.detach()),text=float(out.lm_loss.detach()),mask=float(out.mask_loss.detach())))
    if cuda:torch.cuda.synchronize()
    report=dict(seed=int(torch.initial_seed()),description_supervision=description_supervision,learning_rate=learning_rate,steps_completed=steps,samples_used=min(steps,len(samples)),losses=losses,
        elapsed_seconds=time.monotonic()-start,trainable_parameters=sum(trainable.values()),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,scientific_acceptance=False)
    save_delta(model,output,base_identity=base_identity,train_config=config,report=report)
    model.eval()
    session.identity={'kind':'geo_groundrag_delta','base_identity':base_identity,'delta_manifest_sha256':sha256_file(Path(output)/'manifest.json')}
    return report


def train_bounded(session,samples,output: Path, *, base_identity: dict,steps=2,timeout_seconds=600,**kwargs):
    """The authorized diagnostic budget; long-run entry is a separate explicit choice."""
    if not 1<=steps<=2 or not 1<=len(samples)<=8 or not 0<timeout_seconds<=600:
        raise ValueError('Bounded authorization: <=8 samples, <=2 updates, <=600 seconds')
    return train_joint(session,samples,output,base_identity=base_identity,steps=steps,
                       timeout_seconds=timeout_seconds,**kwargs)
