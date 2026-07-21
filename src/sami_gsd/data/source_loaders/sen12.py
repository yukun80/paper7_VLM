"""Single-time, event-nearest Sen12Landslides canonical source loader."""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any

from sami_gsd.contracts.canonical import SourceIdentity
from sami_gsd.contracts.config import SourceConfig
from sami_gsd.data.materialize import SourceModalityInput, SpatialParentInput
from sami_gsd.utilities.artifacts import canonical_json_bytes, sha256_bytes, sha256_file


SEN12_LOADER_VERSION = "sami_sen12_single_event_nearest_v2_event_round_robin"
MAX_EVENT_OFFSET_DAYS = 30
MAX_CROSS_MODALITY_SPAN_DAYS = 30
_S2_PATTERN = re.compile(r"^(?P<event>.+)_s2_(?P<sample>\d+)\.nc$")
_S2_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")


class Sen12LoadingError(ValueError):
    """Raised when one Sen12 record cannot satisfy the single-time policy."""


def _dependencies() -> tuple[Any, Any]:
    """Load declared NetCDF/NumPy data dependencies."""

    try:
        import netCDF4
        import numpy as np
    except ImportError as error:  # pragma: no cover - minimal installs
        raise Sen12LoadingError("Sen12 loading requires the sami-groundsegdesc[data] extra") from error
    return netCDF4, np


def _true_attribute(value: Any) -> bool:
    """Parse the source's string/bool annotation flag without truthiness bugs."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise Sen12LoadingError(f"invalid annotated attribute: {value!r}")


def _nearest_event_index(dataset: Any, *, event_date: dt.date) -> tuple[int, dt.datetime, int]:
    """Select exactly one acquisition nearest the event date, never a pre/post pair."""

    netCDF4, _ = _dependencies()
    variable = dataset.variables.get("time")
    if variable is None or "units" not in variable.ncattrs():
        raise Sen12LoadingError("Sen12 time coordinate lacks units")
    calendar = variable.getncattr("calendar") if "calendar" in variable.ncattrs() else "standard"
    dates = netCDF4.num2date(variable[:], units=variable.getncattr("units"), calendar=calendar)
    event_datetime = dt.datetime.combine(event_date, dt.time.min)
    normalized = [
        dt.datetime(value.year, value.month, value.day, value.hour, value.minute, value.second)
        for value in dates
    ]
    offsets = [abs((value - event_datetime).days) for value in normalized]
    index = min(range(len(normalized)), key=lambda item: (offsets[item], normalized[item], item))
    if offsets[index] > MAX_EVENT_OFFSET_DAYS:
        raise Sen12LoadingError("nearest acquisition is outside the frozen event window")
    return index, normalized[index], offsets[index]


def _event_date_candidates(value: str) -> tuple[dt.date, ...]:
    """Parse one or more comma-separated source event dates deterministically."""

    candidates: list[dt.date] = []
    for token in value.split(","):
        stripped = token.strip()
        # NetCDF attributes in the local corpus encode a missing date both as an empty
        # string and as the literal text "None". Both mean that this record cannot
        # establish a contemporaneous tuple; they are not source-wide parse failures.
        if not stripped or stripped.casefold() in {"none", "null", "nan"}:
            continue
        try:
            candidates.append(dt.date.fromisoformat(stripped))
        except ValueError as error:
            raise Sen12LoadingError(f"invalid Sen12 event_date token: {stripped!r}") from error
    ordered = tuple(sorted(set(candidates)))
    return ordered


def _grid(dataset: Any) -> tuple[str, tuple[float, float, float, float, float, float]]:
    """Read exact CRS and GDAL transform from the source spatial_ref variable."""

    spatial_ref = dataset.variables.get("spatial_ref")
    if spatial_ref is None or "GeoTransform" not in spatial_ref.ncattrs():
        raise Sen12LoadingError("Sen12 spatial_ref lacks GeoTransform")
    values = tuple(float(value) for value in str(spatial_ref.getncattr("GeoTransform")).split())
    if len(values) != 6:
        raise Sen12LoadingError("Sen12 GeoTransform must have six values")
    crs = dataset.getncattr("crs") if "crs" in dataset.ncattrs() else None
    if not isinstance(crs, str) or not crs:
        raise Sen12LoadingError("Sen12 global CRS attribute is missing")
    return crs, values


def _slice_hwc(dataset: Any, names: tuple[str, ...], index: int) -> tuple[Any, Any]:
    """Read named time slices, transpose source (x,y) to canonical (H=y,W=x), and return validity."""

    _, np = _dependencies()
    arrays: list[Any] = []
    valid: Any | None = None
    for name in names:
        if name not in dataset.variables:
            raise Sen12LoadingError(f"Sen12 variable is missing: {name}")
        variable = dataset.variables[name]
        if variable.dimensions != ("time", "x", "y"):
            raise Sen12LoadingError(f"Sen12 variable has unexpected dimensions: {name}:{variable.dimensions}")
        masked = np.ma.asarray(variable[index]).T
        channel_valid = ~np.ma.getmaskarray(masked) & np.isfinite(masked.filled(np.nan))
        valid = channel_valid if valid is None else valid & channel_valid
        arrays.append(masked.filled(0.0).astype("float32", copy=False))
    assert valid is not None
    return np.stack(arrays, axis=2), valid.astype("uint8")


def _mask_slice(dataset: Any, index: int) -> Any:
    """Read one official binary mask and reject non-binary values."""

    _, np = _dependencies()
    variable = dataset.variables.get("MASK")
    if variable is None or variable.dimensions != ("time", "x", "y"):
        raise Sen12LoadingError("Sen12 MASK must use (time,x,y)")
    mask = np.ma.asarray(variable[index]).T
    if np.ma.getmaskarray(mask).any():
        raise Sen12LoadingError("Sen12 official mask contains masked pixels")
    values = mask.filled(0)
    if not np.all((values == 0) | (values == 1)):
        raise Sen12LoadingError("Sen12 official mask is not binary")
    return values.astype("uint8", copy=False)


def _s2_valid(dataset: Any, index: int, base_valid: Any) -> Any:
    """Exclude SCL nodata/saturation/shadow/cloud/cirrus/snow classes."""

    _, np = _dependencies()
    variable = dataset.variables.get("SCL")
    if variable is None or variable.dimensions != ("time", "x", "y"):
        raise Sen12LoadingError("Sen12 SCL must use (time,x,y)")
    scl = np.ma.asarray(variable[index]).T
    invalid_classes = np.isin(scl.filled(0), (0, 1, 3, 8, 9, 10, 11))
    return (base_valid.astype(bool) & ~np.ma.getmaskarray(scl) & ~invalid_classes).astype("uint8")


def _load_triplet(
    source: SourceConfig,
    *,
    source_root: Path,
    s2_path: Path,
    event: str,
    sample: str,
) -> SpatialParentInput | None:
    """Load one annotated S2/ASC/DSC triplet under the frozen single-time rule."""

    netCDF4, np = _dependencies()
    asc_path = source_root / "s1asc" / f"{event}_s1asc_{sample}.nc"
    dsc_path = source_root / "s1dsc" / f"{event}_s1dsc_{sample}.nc"
    if not asc_path.is_file() or not dsc_path.is_file():
        return None
    with (
        netCDF4.Dataset(s2_path, mode="r") as s2,
        netCDF4.Dataset(asc_path, mode="r") as asc,
        netCDF4.Dataset(dsc_path, mode="r") as dsc,
    ):
        flags = tuple(_true_attribute(dataset.getncattr("annotated")) for dataset in (s2, asc, dsc))
        if not all(flags):
            return None
        event_dates = tuple(str(dataset.getncattr("event_date")) for dataset in (s2, asc, dsc))
        if len(set(event_dates)) != 1 or not event_dates[0]:
            raise Sen12LoadingError(f"paired event dates disagree: {event}/{sample}")
        candidate_selections: list[
            tuple[tuple[int, int, dt.date], dt.date, tuple[tuple[int, dt.datetime, int], ...]]
        ] = []
        for candidate_date in _event_date_candidates(event_dates[0]):
            try:
                candidate = tuple(
                    _nearest_event_index(dataset, event_date=candidate_date)
                    for dataset in (s2, asc, dsc)
                )
            except Sen12LoadingError:
                continue
            candidate_acquisitions = [selection[1] for selection in candidate]
            if (
                max(candidate_acquisitions) - min(candidate_acquisitions)
            ).days > MAX_CROSS_MODALITY_SPAN_DAYS:
                continue
            offsets = tuple(selection[2] for selection in candidate)
            candidate_selections.append(
                ((max(offsets), sum(offsets), candidate_date), candidate_date, candidate)
            )
        if not candidate_selections:
            # 该记录不满足冻结的同期窗口，属于可审计的样本级技术排除，而不是整源解析失败。
            return None
        _, event_date, selections = min(candidate_selections, key=lambda item: item[0])
        acquisition_dates = [selection[1] for selection in selections]
        grids = tuple(_grid(dataset) for dataset in (s2, asc, dsc))
        if len({grid[0] for grid in grids}) != 1 or any(
            not all(abs(left - right) <= 1e-12 for left, right in zip(grids[0][1], grid[1], strict=True))
            for grid in grids[1:]
        ):
            raise Sen12LoadingError(f"paired source grids disagree: {event}/{sample}")

        s2_array, s2_valid = _slice_hwc(s2, _S2_BANDS, selections[0][0])
        s2_valid = _s2_valid(s2, selections[0][0], s2_valid)
        # S2 是该 source 的确定性参考视图。全云/全 nodata 的时间片无法提供
        # duplicate、mask 或像素监督的有效画布，属于样本级技术排除。
        if not np.any(s2_valid):
            return None
        asc_array, asc_valid = _slice_hwc(asc, ("VV", "VH"), selections[1][0])
        dsc_array, dsc_valid = _slice_hwc(dsc, ("VV", "VH"), selections[2][0])
        dem_array, dem_valid = _slice_hwc(s2, ("DEM",), selections[0][0])
        masks = tuple(_mask_slice(dataset, selection[0]) for dataset, selection in zip((s2, asc, dsc), selections, strict=True))
        if not all(np.array_equal(masks[0], value) for value in masks[1:]):
            raise Sen12LoadingError(f"paired official masks disagree: {event}/{sample}")

    s2_hash = sha256_file(s2_path)
    asc_hash = sha256_file(asc_path)
    dsc_hash = sha256_file(dsc_path)
    logical_paths = (
        f"{source.provenance.source_root}/s2/{s2_path.name}",
        f"{source.provenance.source_root}/s1asc/{asc_path.name}",
        f"{source.provenance.source_root}/s1dsc/{dsc_path.name}",
    )
    crs, geotransform = grids[0]
    gsd = abs(geotransform[1])
    modalities = (
        SourceModalityInput(
            modality_id="s2_optical",
            family="multispectral",
            sensor="Sentinel-2",
            product_type="single_event_nearest_multispectral_slice",
            band_names=_S2_BANDS,
            array=s2_array,
            valid=s2_valid,
            source_logical_path=logical_paths[0],
            source_sha256=s2_hash,
            units="DN",
            signed=False,
            sign_convention=None,
            native_gsd_m=gsd,
            acquisition_time=acquisition_dates[0].isoformat() + "Z",
            crs=crs,
            geotransform=geotransform,
        ),
        SourceModalityInput(
            modality_id="s1_ascending",
            family="sar",
            sensor="Sentinel-1",
            product_type="single_event_nearest_backscatter_slice",
            band_names=("VV", "VH"),
            array=asc_array,
            valid=asc_valid,
            source_logical_path=logical_paths[1],
            source_sha256=asc_hash,
            units="dB",
            signed=True,
            sign_convention="logarithmic backscatter magnitude; sign is not a motion direction",
            native_gsd_m=gsd,
            orbit="ascending",
            acquisition_time=acquisition_dates[1].isoformat() + "Z",
            crs=crs,
            geotransform=geotransform,
        ),
        SourceModalityInput(
            modality_id="s1_descending",
            family="sar",
            sensor="Sentinel-1",
            product_type="single_event_nearest_backscatter_slice",
            band_names=("VV", "VH"),
            array=dsc_array,
            valid=dsc_valid,
            source_logical_path=logical_paths[2],
            source_sha256=dsc_hash,
            units="dB",
            signed=True,
            sign_convention="logarithmic backscatter magnitude; sign is not a motion direction",
            native_gsd_m=gsd,
            orbit="descending",
            acquisition_time=acquisition_dates[2].isoformat() + "Z",
            crs=crs,
            geotransform=geotransform,
        ),
        SourceModalityInput(
            modality_id="dem",
            family="dem",
            sensor="Copernicus-WorldDEM-30",
            product_type="elevation",
            band_names=("DEM",),
            array=dem_array,
            valid=dem_valid,
            source_logical_path=logical_paths[0],
            source_sha256=s2_hash,
            units="m",
            signed=False,
            sign_convention=None,
            native_gsd_m=gsd,
            acquisition_time=None,
            crs=crs,
            geotransform=geotransform,
        ),
    )
    source_record_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "loader_version": SEN12_LOADER_VERSION,
                "paths": logical_paths,
                "hashes": (s2_hash, asc_hash, dsc_hash),
                "selection": [
                    {"index": selection[0], "acquisition": selection[1].isoformat(), "event_offset_days": selection[2]}
                    for selection in selections
                ],
            }
        )
    )
    return SpatialParentInput(
        parent_id=f"sen12-{event}-{sample}",
        source=SourceIdentity(
            dataset="Sen12Landslides",
            record_id=f"{event}/{sample}",
            scene_id=f"{event}-{sample}",
            event_id=f"{event}:{event_date.isoformat()}",
            region_id=event,
            source_group_id=f"sen12/{event}/{sample}",
        ),
        reference_modality_id="s2_optical",
        modalities=modalities,
        global_mask=masks[0],
        global_mask_origin="official",
        referring_regions=(),
        source_registry_key=source.source_key,
        source_record_sha256=source_record_hash,
        annotation_status="gold",
    )


def load_sen12_parents(
    source: SourceConfig,
    *,
    source_root: Path,
    limit: int | None,
) -> tuple[SpatialParentInput, ...]:
    """Load annotated records in deterministic event round-robin order.

    Small construction must not be an accidental prefix of one lexically first
    disaster. Interleaving filename groups keeps the bounded set event-diverse
    before parent-level split assignment while preserving read-only decoding.
    """

    s2_root = source_root / "s2"
    if not s2_root.is_dir():
        raise Sen12LoadingError("Sen12 s2 directory is missing")
    paths_by_event: dict[str, list[tuple[Path, str]]] = {}
    for s2_path in sorted(s2_root.glob("*.nc"), key=lambda path: path.name):
        match = _S2_PATTERN.match(s2_path.name)
        if match is None:
            raise Sen12LoadingError(f"unexpected Sen12 S2 filename: {s2_path.name}")
        paths_by_event.setdefault(match.group("event"), []).append((s2_path, match.group("sample")))

    parents: list[SpatialParentInput] = []
    offsets = {event: 0 for event in paths_by_event}
    active_events = sorted(paths_by_event)
    while active_events and (limit is None or len(parents) < limit):
        following_events: list[str] = []
        for event in active_events:
            rows = paths_by_event[event]
            offset = offsets[event]
            if offset >= len(rows):
                continue
            s2_path, sample = rows[offset]
            offsets[event] = offset + 1
            if offsets[event] < len(rows):
                following_events.append(event)
            parent = _load_triplet(
                source,
                source_root=source_root,
                s2_path=s2_path,
                event=event,
                sample=sample,
            )
            if parent is not None:
                parents.append(parent)
                if limit is not None and len(parents) >= limit:
                    break
        active_events = following_events
    return tuple(parents)


__all__ = [
    "MAX_CROSS_MODALITY_SPAN_DAYS",
    "MAX_EVENT_OFFSET_DAYS",
    "SEN12_LOADER_VERSION",
    "Sen12LoadingError",
    "load_sen12_parents",
]
