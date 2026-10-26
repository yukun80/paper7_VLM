"""Pinned official Lens model with project-owned shared auxiliary injection."""
from contextlib import contextmanager
from functools import wraps
from collections.abc import Mapping, Sequence
import torch
from xiaomi_lens.model.modeling_xiaomi_lens import XiaomiLensQwen3_5ForConditionalGeneration
from .contracts import AuxiliaryObservation, validate_mask_binding
from .fusion import AuxiliaryFusion

class GeoGroundModel(XiaomiLensQwen3_5ForConditionalGeneration):
    def __init__(self, config):
        config.lm_loss_impl = "torch"
        super().__init__(config)
        self.auxiliary_fusion = AuxiliaryFusion(config.vision_config.hidden_size)
        self._aux_context = None
        self.model.visual.register_forward_hook(self._fuse_visual, with_kwargs=True)

    @contextmanager
    def auxiliary_context(self, observations: Sequence[Mapping[str, AuxiliaryObservation]], image_sizes: Sequence[tuple[int,int]]):
        if self._aux_context is not None:
            raise RuntimeError("A model session is sequential; nested/concurrent requests are forbidden")
        self._aux_context = (observations, image_sizes)
        try:
            yield
        finally:
            self._aux_context = None

    def _fuse_visual(self, module, args, kwargs, output):
        if self._aux_context is None:
            return output
        observations, image_sizes = self._aux_context
        grid = kwargs.get("grid_thw", args[1] if len(args)>1 else None)
        if grid is None:
            raise ValueError("Visual grid is required")
        fused = self.auxiliary_fusion(output.last_hidden_state, grid, observations, image_sizes,
                                      self.config.vision_config.spatial_merge_size)
        output.last_hidden_state = fused
        # Both branches now consume exactly this fused representation.
        output.pooler_output = module.merger(fused)
        return output

    @wraps(XiaomiLensQwen3_5ForConditionalGeneration.forward)
    def forward(self, *args, **kwargs):
        positions = kwargs.get("mask_positions")
        if positions is not None and positions.numel():
            ids = kwargs.get("input_ids")
            indices = kwargs.get("mask_image_index")
            if ids is None or indices is None:
                raise ValueError("Every mask requires explicit input IDs and image binding")
            counts = (ids == self.config.vision_start_token_id).sum(1).tolist()
            validate_mask_binding(ids, positions, indices, counts, self.config.mask_token_id, kwargs.get("mask_valid_mask"))
            targets = kwargs.get("mask_targets")
            if targets is not None:
                if targets.ndim != 4 or targets.shape[:2] != positions.shape:
                    raise ValueError("Mask token/target count mismatch")
                if not torch.isfinite(targets).all() or not ((targets == 0) | (targets == 1)).all():
                    raise ValueError("Mask supervision must be finite and binary")
                pixel_valid = kwargs.get("mask_pixel_valid")
                slot_valid = kwargs.get("mask_valid_mask")
                if pixel_valid is not None:
                    if pixel_valid.shape != targets.shape or not ((pixel_valid == 0) | (pixel_valid == 1)).all():
                        raise ValueError("Invalid supervision validity map")
                    if slot_valid is not None and pixel_valid[~slot_valid.bool()].any():
                        raise ValueError("Padded mask slots cannot supervise pixels")
        return super().forward(*args, **kwargs)


def tiny_model() -> GeoGroundModel:
    """Actual Qwen3.5 + official Lens decoder, randomly initialized; never a scientific model."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    config = Qwen3_5Config(
        text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, layer_types=["linear_attention", "full_attention"],
                         linear_key_head_dim=8, linear_value_head_dim=8,
                         linear_num_key_heads=4, linear_num_value_heads=4,
                         max_position_embeddings=512, rope_parameters={"rope_type":"default", "rope_theta":10000.0,"mrope_section":[1,1,2],"partial_rotary_factor":1.0}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                           patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
                           out_hidden_size=32, num_position_embeddings=64),
        image_token_id=60, vision_start_token_id=61, vision_end_token_id=62,
        mask_token_id=63, mask_decoder_channels=16, mask_decoder_embed_dim=8,
        pad_token_id=0, eos_token_id=2,
    )
    config._attn_implementation = "sdpa"
    return GeoGroundModel(config)
