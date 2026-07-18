"""Pure construction of the only accepted formal D0 training command."""

from __future__ import annotations

from pathlib import Path
import shlex
from typing import Any


D0_TRAINING_LAUNCH_PROTOCOL = (
    "qpsalm_d0_training_launch_v2_exact_command_bound"
)


def build_d0_training_launch(
    *,
    python_executable: str,
    config_path: str | Path,
    config_sha256: str,
    seed: int,
    device_name: str,
    d_minus_one_gate: str | Path,
    output_dir: str | Path,
    preflight_report: str | Path,
) -> dict[str, Any]:
    """Return canonical argv plus the shell rendering for formal D0."""

    resolved_config = Path(config_path).resolve(strict=False)
    resolved_output = Path(output_dir).resolve(strict=False)
    resolved_report = Path(preflight_report).resolve(strict=False)
    argv = [
        str(python_executable),
        "-B",
        "-m",
        "qpsalm_seg.cli.segdesc",
        "train",
        "--config",
        str(resolved_config),
        "--stage",
        "mmrs_caption",
        "--seed",
        str(int(seed)),
        "--device",
        str(device_name),
        "--d-minus-one-gate",
        str(d_minus_one_gate),
        "--output-dir",
        str(resolved_output),
        "--d0-preflight-report",
        str(resolved_report),
    ]
    return {
        "protocol": D0_TRAINING_LAUNCH_PROTOCOL,
        "pythonpath": "SEG_Multi-Source_Landslides",
        "argv": argv,
        "resolved_config": str(resolved_config),
        "resolved_config_sha256": str(config_sha256),
        "output_dir": str(resolved_output),
        "device": str(device_name),
        "command": (
            "PYTHONPATH=SEG_Multi-Source_Landslides " + shlex.join(argv)
        ),
        "unique": True,
    }
