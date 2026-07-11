#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensor-aware multi-view renderer 测试。

用途：验证 S2 真/假彩色、signed InSAR 渲染和 visual evidence view 消融。
推荐运行命令：PYTHONPATH=SEG_Multi-Source_Landslides python -m unittest
SEG_Multi-Source_Landslides/tests/test_renderer.py -v
主要输入：代码内构造的合成 ModalityInstance。
主要输出：unittest 终端结果。
写入行为：不写文件。
所属流程：Qwen 多源视觉证据渲染回归测试。
"""

from __future__ import annotations

import unittest

import torch

from qpsalm_seg.cli.cache_qwen_visual_evidence import apply_visual_ablation
from qpsalm_seg.rendering import render_sensor_views
from qpsalm_seg.schema import ModalityInstance


class RendererTest(unittest.TestCase):
    def test_s2_and_signed_insar_views(self) -> None:
        s2 = ModalityInstance(
            name="multispectral",
            family="multispectral",
            sensor="sentinel2",
            band_names=("B02", "B03", "B04", "B08", "B11", "B12"),
            orbit="unknown",
            image=torch.rand((6, 16, 16)),
            valid_mask=torch.ones((1, 16, 16)),
            native_gsd_m=10.0,
            aligned_gsd_m=10.0,
        )
        insar_image = torch.linspace(-1.0, 1.0, 256).view(1, 16, 16)
        insar = ModalityInstance(
            name="insar_vel",
            family="deformation",
            sensor="insar",
            band_names=("insar_velocity",),
            orbit="unknown",
            image=insar_image,
            valid_mask=torch.ones((1, 16, 16)),
            native_gsd_m=10.0,
            aligned_gsd_m=10.0,
        )
        views = render_sensor_views([s2, insar], size=32)
        names = {view.name for view in views}
        self.assertIn("multispectral_true_color", names)
        self.assertIn("multispectral_false_color", names)
        signed = next(view for view in views if view.name == "insar_vel_signed").image
        self.assertGreater(float(signed[2, 8, 8]), float(signed[0, 8, 8]))
        self.assertGreater(float(signed[0, 24, 24]), float(signed[2, 24, 24]))

    def test_view_shuffle_and_removal_are_explicit(self) -> None:
        samples = []
        for index in range(3):
            modality = ModalityInstance(
                name=f"optical_{index}",
                family="optical",
                sensor="rgb",
                band_names=("R", "G", "B"),
                orbit="unknown",
                image=torch.rand((3, 12, 12), generator=torch.Generator().manual_seed(index)),
                valid_mask=torch.ones((1, 12, 12)),
                native_gsd_m=0.5,
                aligned_gsd_m=0.5,
            )
            samples.append(
                {
                    "lookup_key": f"sample-{index}",
                    "views": render_sensor_views([modality], size=16),
                    "metadata": {},
                }
            )
        ablation = apply_visual_ablation(samples, [], True, 7)
        self.assertTrue(ablation["shuffle_views_across_samples"])
        for sample in samples:
            self.assertNotEqual(sample["lookup_key"], sample["metadata"]["visual_source_lookup_key"])


if __name__ == "__main__":
    unittest.main()
