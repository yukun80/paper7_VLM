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

from qpsalm_seg.rendering import render_sensor_views
from qpsalm_seg.schema import ModalityInstance


class RendererTest(unittest.TestCase):
    @staticmethod
    def _instance(**overrides) -> ModalityInstance:
        values = {
            "name": "modality", "family": "terrain", "sensor": "generic_dem",
            "product_type": "slope", "band_names": ("SLOPE",),
            "band_metadata": ({"native_gsd_m": 10.0, "signed": False},),
            "orbit": "unknown", "units": "degree", "signed": False,
            "image": torch.rand((1, 8, 8)), "valid_mask": torch.ones((1, 8, 8)),
            "native_gsd_m": 10.0, "aligned_gsd_m": 10.0,
        }
        values.update(overrides)
        return ModalityInstance(**values)

    def test_s2_and_signed_insar_views(self) -> None:
        s2 = ModalityInstance(
            name="multispectral",
            family="multispectral",
            sensor="sentinel2",
            product_type="surface_reflectance",
            band_names=("B02", "B03", "B04", "B08", "B11", "B12"),
            band_metadata=tuple({"native_gsd_m": 10.0, "signed": False} for _ in range(6)),
            orbit="unknown",
            units="reflectance",
            signed=False,
            image=torch.rand((6, 16, 16)),
            valid_mask=torch.ones((1, 16, 16)),
            native_gsd_m=10.0,
            aligned_gsd_m=10.0,
        )
        insar_image = torch.linspace(-1.0, 1.0, 256).view(1, 16, 16)
        insar = ModalityInstance(
            name="insar_vel",
            family="deformation",
            sensor="generic_insar",
            product_type="los_velocity",
            band_names=("INSAR_VELOCITY",),
            band_metadata=({"native_gsd_m": 10.0, "signed": True},),
            orbit="unknown",
            units="mm/year",
            signed=True,
            image=insar_image,
            valid_mask=torch.ones((1, 16, 16)),
            native_gsd_m=10.0,
            aligned_gsd_m=10.0,
        )
        views = render_sensor_views([s2, insar], size=32)
        names = {view.name for view in views}
        self.assertIn("multispectral_true_color", names)
        self.assertIn("multispectral_false_color", names)
        signed = next(view for view in views if view.name == "insar_vel_signed_fixed_scale").image
        self.assertGreater(float(signed[2, 8, 8]), float(signed[0, 8, 8]))
        self.assertGreater(float(signed[0, 24, 24]), float(signed[2, 24, 24]))

    def test_removed_modality_produces_no_view(self) -> None:
        optical = ModalityInstance(
            name="optical_rgb", family="optical", sensor="generic_rgb", product_type="rgb",
            band_names=("R", "G", "B"),
            band_metadata=tuple({"native_gsd_m": 0.5, "signed": False} for _ in range(3)),
            orbit="unknown", units="reflectance", signed=False,
            image=torch.rand((3, 12, 12)), valid_mask=torch.ones((1, 12, 12)),
            native_gsd_m=0.5, aligned_gsd_m=0.5,
        )
        names = {view.name for view in render_sensor_views([optical], size=16)}
        self.assertEqual(names, {"optical_rgb_true_color"})

    def test_terrain_product_is_not_reinterpreted_as_elevation(self) -> None:
        slope = self._instance(name="slope")
        view = render_sensor_views([slope], size=16)[0]
        self.assertIn("no elevation derivatives", view.description)
        self.assertTrue(torch.allclose(view.image[0], view.image[1]))
        self.assertTrue(torch.allclose(view.image[1], view.image[2]))

    def test_sar_difference_label_and_nodata_gray_are_physical(self) -> None:
        image = torch.stack([torch.full((8, 8), 0.8), torch.full((8, 8), 0.3)])
        valid = torch.ones((1, 8, 8))
        valid[:, :2] = 0
        sar = self._instance(
            name="sar_asc", family="sar", sensor="sentinel1", product_type="sar_backscatter",
            band_names=("VV", "VH"),
            band_metadata=tuple({"native_gsd_m": 10.0, "signed": False} for _ in range(2)),
            orbit="ascending", units="dB", image=image, valid_mask=valid,
        )
        view = render_sensor_views([sar], size=16)[0]
        self.assertIn("difference", view.description.lower())
        self.assertNotIn("ratio", view.description.lower())
        self.assertTrue(torch.allclose(view.image[:, :4], torch.full_like(view.image[:, :4], 0.5)))

    def test_unknown_multispectral_band_order_is_rejected_in_strict_mode(self) -> None:
        unknown = self._instance(
            name="unknown_s2", family="multispectral", sensor="sentinel2",
            product_type="surface_reflectance", band_names=("X1", "X2", "X3"),
            band_metadata=tuple({"native_gsd_m": 10.0, "signed": False} for _ in range(3)),
            units="reflectance", image=torch.rand((3, 8, 8)),
        )
        with self.assertRaises(ValueError):
            render_sensor_views([unknown], size=16, strict=True)

    def test_nodata_outlier_does_not_flatten_valid_optical_contrast(self) -> None:
        image = torch.zeros((3, 8, 8))
        image[:, :, 4:] = 100.0
        image[:, 0, 0] = -9999.0
        valid = torch.ones((1, 8, 8))
        valid[:, 0, 0] = 0
        optical = self._instance(
            name="optical", family="optical", sensor="generic_rgb", product_type="rgb",
            band_names=("R", "G", "B"),
            band_metadata=tuple({"native_gsd_m": 1.0, "signed": False} for _ in range(3)),
            units="digital_number", image=image, valid_mask=valid,
        )
        view = render_sensor_views([optical], size=8)[0]
        self.assertLess(float(view.image[:, 4, 1].mean()), 0.05)
        self.assertGreater(float(view.image[:, 4, 6].mean()), 0.95)
        self.assertTrue(torch.allclose(view.image[:, 0, 0], torch.full((3,), 0.5)))


if __name__ == "__main__":
    unittest.main()
