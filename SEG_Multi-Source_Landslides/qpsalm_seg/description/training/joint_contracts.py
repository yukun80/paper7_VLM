#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable M7 protocol identifiers shared by training and evaluation."""

from __future__ import annotations


JOINT_RUN_PROTOCOL = "qpsalm_segdesc_joint_v7_strict_json_finite"
JOINT_INITIALIZATION_PROTOCOL = (
    "qpsalm_segdesc_joint_initialization_v4_run_completion_bound"
)
JOINT_PROGRESS_PROTOCOL = (
    "qpsalm_segdesc_joint_progress_v3_parent_population_list_bound"
)
JOINT_LOADER_BINDING_PROTOCOL = "qpsalm_segdesc_joint_loader_binding_v1"
JOINT_LOADER_CURSOR_PROTOCOL = "qpsalm_segdesc_joint_loader_cursor_v1"
JOINT_TASKS = ("segmentation", "global_caption", "region_description")
JOINT_LOADER_SEED_OFFSETS = {
    "segmentation": 710_011,
    "global_caption": 720_013,
    "region_description": 730_019,
}

