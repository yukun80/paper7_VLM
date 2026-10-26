"""Demo-only input previews；只渲染现有 Benchmark 数值，不参与模型输入。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from oa_groundrag.grounding.evidence import render_evidence_image

from .i18n import DEFAULT_LOCALE, preview_caption


DISPLAY_TRANSFORM = "finite_valid_linear_minmax_to_uint8_grayscale"


class DemoPreviewError(RuntimeError):
    """Benchmark preview 的 shape、channel metadata 或 validity 合同失败。"""


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class InputChannelPreview:
    """单个真实 channel 的确定性展示；``image`` 是派生视图而非科学数据。"""

    sample_id: str
    modality: str
    channel_index: int
    channel_name: str
    image: Image.Image
    channel_valid: bool
    valid_pixel_count: int
    total_pixel_count: int
    raw_min: float | None
    raw_max: float | None
    unit: str | None
    sign_convention: str | None
    is_auxiliary: bool

    @property
    def valid_fraction(self) -> float:
        if self.total_pixel_count <= 0:
            return 0.0
        return float(self.valid_pixel_count / self.total_pixel_count)

    @property
    def caption(self) -> str:
        """Viewer artifact 的稳定原始 caption；不随 UI locale 改写。"""

        role = "Spatial Expert Input Preview"
        if self.is_auxiliary:
            boundary = "Not formal MLLM grounded input in current P0"
        else:
            boundary = "Raw optical channel preview; Full Optical RGB is the P0 visual input"
        return (
            f"{role} | {self.modality} / {self.channel_name} | "
            f"valid={self.valid_fraction:.3f}\n{boundary}"
        )

    def gallery_value(self, locale: str = DEFAULT_LOCALE) -> tuple[Image.Image, str]:
        return self.image, preview_caption(locale, self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "modality": self.modality,
            "channel_index": self.channel_index,
            "channel_name": self.channel_name,
            "channel_valid": self.channel_valid,
            "valid_pixel_count": self.valid_pixel_count,
            "total_pixel_count": self.total_pixel_count,
            "valid_fraction": self.valid_fraction,
            "raw_min": self.raw_min,
            "raw_max": self.raw_max,
            "unit": self.unit,
            "sign_convention": self.sign_convention,
            "display_transform": DISPLAY_TRANSFORM,
            "invalid_pixels_rendered_as_black": True,
            "spatial_expert_input_preview": True,
            "formal_mllm_grounded_input": False,
            "is_auxiliary": self.is_auxiliary,
        }


def _optional_text(contract: Mapping[str, Any], key: str) -> str | None:
    value = contract.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DemoPreviewError(f"{key} 必须是非空字符串或 null")
    return value.strip()


def _channel_previews(
    *,
    sample_id: str,
    modality: str,
    values: Any,
    pixel_valid: Any,
    channel_valid: Any,
    channel_names: Sequence[str],
    expected_channel_names: Sequence[str],
    contract: Mapping[str, Any],
    is_auxiliary: bool,
) -> tuple[InputChannelPreview, ...]:
    array = _numpy(values)
    validity = _numpy(pixel_valid)
    channels = _numpy(channel_valid)
    names = tuple(str(value) for value in channel_names)
    expected = tuple(str(value) for value in expected_channel_names)
    if (
        array.ndim != 3
        or validity.shape != array.shape
        or channels.shape != (array.shape[0],)
        or len(names) != array.shape[0]
        or names != expected
    ):
        raise DemoPreviewError(
            f"{sample_id}:{modality} channel/value/validity metadata 不匹配"
        )
    unit = _optional_text(contract, "unit")
    sign = _optional_text(contract, "sign_convention")
    previews: list[InputChannelPreview] = []
    for index, name in enumerate(names):
        valid_channel = bool(channels[index])
        effective = (
            validity[index].astype(bool)
            & valid_channel
            & np.isfinite(array[index])
        )
        finite_values = np.asarray(array[index][effective], dtype=np.float64)
        image = render_evidence_image(
            array[index],
            validity=effective,
            location=f"demo_preview:{sample_id}:{modality}:{index}",
        )
        previews.append(InputChannelPreview(
            sample_id=sample_id,
            modality=modality,
            channel_index=index,
            channel_name=name,
            image=image,
            channel_valid=valid_channel,
            valid_pixel_count=int(effective.sum()),
            total_pixel_count=int(effective.size),
            raw_min=None if finite_values.size == 0 else float(finite_values.min()),
            raw_max=None if finite_values.size == 0 else float(finite_values.max()),
            unit=unit,
            sign_convention=sign,
            is_auxiliary=is_auxiliary,
        ))
    return tuple(previews)


def build_input_channel_previews(
    sample: Mapping[str, Any],
    *,
    expected_optical_channels: Sequence[str],
    expected_auxiliary_channels: Mapping[str, Sequence[str]],
    modality_contracts: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[InputChannelPreview, ...], tuple[InputChannelPreview, ...]]:
    """从同一个 raw Benchmark sample 构建 optical 与 auxiliary channel gallery。"""

    sample_id = str(sample.get("sample_id", ""))
    if not sample_id:
        raise DemoPreviewError("Benchmark sample 缺少 sample_id")
    optical = _channel_previews(
        sample_id=sample_id,
        modality="optical",
        values=sample["optical"],
        pixel_valid=sample["optical_pixel_valid"],
        channel_valid=sample["optical_channel_valid"],
        channel_names=sample["optical_channel_names"],
        expected_channel_names=expected_optical_channels,
        contract={},
        is_auxiliary=False,
    )
    auxiliaries = sample.get("auxiliaries")
    if not isinstance(auxiliaries, Mapping):
        raise DemoPreviewError(f"{sample_id}: auxiliaries 必须是 mapping")
    if set(auxiliaries) != set(expected_auxiliary_channels):
        raise DemoPreviewError(
            f"{sample_id}: raw sample 与 index auxiliary registry 不一致"
        )
    rendered: list[InputChannelPreview] = []
    for modality in sorted(auxiliaries):
        value = auxiliaries[modality]
        if not isinstance(value, Mapping):
            raise DemoPreviewError(f"{sample_id}:{modality} auxiliary payload 非法")
        contract = modality_contracts.get(modality, {})
        if not isinstance(contract, Mapping):
            raise DemoPreviewError(f"{sample_id}:{modality} index contract 非法")
        rendered.extend(_channel_previews(
            sample_id=sample_id,
            modality=str(modality),
            values=value["values"],
            pixel_valid=value["pixel_valid"],
            channel_valid=value["channel_valid"],
            channel_names=value["channel_names"],
            expected_channel_names=expected_auxiliary_channels[modality],
            contract=contract,
            is_auxiliary=True,
        ))
    return optical, tuple(rendered)
