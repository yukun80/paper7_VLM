"""Stable logical identities for Description Vision Cache v1."""

from __future__ import annotations


def description_cache_key(component: str, parent_sample_id: str) -> str:
    if component not in {"single_image", "multisource_parent"}:
        raise ValueError(f"未知 description cache component={component!r}")
    return f"qdcv1:{component}:{parent_sample_id}"

