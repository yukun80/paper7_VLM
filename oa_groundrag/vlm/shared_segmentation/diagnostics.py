"""Explicit untrained-decoder resource probe; not a Lens checkpoint or a prediction model."""
from pathlib import Path
import torch
from .model import GeoGroundModel
from .session import SharedModelSession


def load_base_for_diagnostics(root: Path, ledger_path: Path, *, device='cuda', max_pixels=1024):
    from transformers import AutoProcessor,AutoConfig
    from oa_groundrag.vlm.backends.assets import verify_model_asset_ledger
    from oa_groundrag.vlm.backends.qwen3_5.model import QWEN35_UNUSED_MTP_TENSORS
    assets=verify_model_asset_ledger(ledger_path,expected_backend='qwen3_5',
        expected_repo_id='Qwen/Qwen3.5-4B',expected_model_root=root)
    processor=AutoProcessor.from_pretrained(str(root),local_files_only=True)
    processor.tokenizer.add_special_tokens({'additional_special_tokens':['<|mask|>']})
    config=AutoConfig.from_pretrained(str(root),local_files_only=True)
    config.mask_token_id=processor.tokenizer.convert_tokens_to_ids('<|mask|>')
    config.lm_loss_impl='torch'
    model,info=GeoGroundModel.from_pretrained(str(root),config=config,local_files_only=True,
        attn_implementation='sdpa',dtype=torch.bfloat16 if device=='cuda' else torch.float32,
        device_map={'':device},output_loading_info=True)
    unknown=set(info.get('unexpected_keys',[]))-set(QWEN35_UNUSED_MTP_TENSORS)
    if unknown or info.get('mismatched_keys') or any(not k.startswith(('mask_decoder.','auxiliary_fusion.')) for k in info.get('missing_keys',[])):
        raise ValueError(f'Unexpected base mapping: {info}')
    model.resize_token_embeddings(max(len(processor.tokenizer),model.get_input_embeddings().weight.shape[0]),mean_resizing=False)
    with torch.no_grad():
        model.get_input_embeddings().weight[config.mask_token_id].copy_(model.get_input_embeddings().weight[:1000].float().mean(0))
    for adapter in model.auxiliary_fusion.adapters.values():
        torch.nn.init.zeros_(adapter[-1].weight);torch.nn.init.zeros_(adapter[-1].bias)
    session=SharedModelSession(model,processor,max_pixels=max_pixels,max_new_tokens=8)
    identity={'kind':'qwen_base_random_decoder_DIAGNOSTIC_ONLY','model_root':str(root),
              'ledger_sha256':assets.ledger_file_sha256,'seed':int(torch.initial_seed()),'segmentation_pretrained':False}
    session.identity=identity
    return session,identity
