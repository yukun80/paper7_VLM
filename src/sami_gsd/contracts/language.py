"""Strict P1 language-source subset contracts."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from sami_gsd.contracts.canonical import (
    LicenseRecord,
    Sha256,
    StrictModel,
    validate_portable_path,
)


class LanguageImageRef(StrictModel):
    """Immutable raw image evidence retained before licensed materialization."""

    logical_path: str
    sha256: Sha256
    native_hw: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]]

    _logical_path_is_portable = field_validator("logical_path")(validate_portable_path)


class LanguageAnswer(StrictModel):
    """One source caption or short phrase with answer-level provenance."""

    answer_id: Annotated[str, Field(min_length=1)]
    text: Annotated[str, Field(min_length=1)]
    annotation_origin: Literal["source_caption", "source_expression"]
    index_logical_path: str
    index_sha256: Sha256

    _index_path_is_portable = field_validator("index_logical_path")(validate_portable_path)


class DescriptionSourceRecord(StrictModel):
    """One frozen selected language record; audit rows may remain unlicensed."""

    schema_version: Literal["sami_description_source_v1"]
    record_id: Annotated[str, Field(min_length=1)]
    source_key: Literal["mmrs_1m", "rsgpt"]
    component: Literal["rsicd", "ucm", "sydney", "nwpu", "rsitmd", "dior_rsvg", "rsicap", "rsieval"]
    source_group_id: Annotated[str, Field(min_length=1)]
    role: Literal["global_caption", "region_short_phrase"]
    split_policy: Literal["train_candidate", "permanent_test_only"]
    image: LanguageImageRef
    answers: tuple[LanguageAnswer, ...]
    normalized_box_xyxy: tuple[float, float, float, float] | None
    license: LicenseRecord
    training_eligible: bool

    @model_validator(mode="after")
    def subset_role_and_license_are_closed(self) -> Self:
        """Reject role drift, test leakage and unlicensed promotion."""

        if not self.answers:
            raise ValueError("description source record requires at least one answer")
        if self.source_key != self.license.source_key:
            raise ValueError("description source/license keys do not match")
        is_region = self.role == "region_short_phrase"
        if is_region != (self.component == "dior_rsvg"):
            raise ValueError("DIOR-RSVG is the sole region-short-phrase component")
        if is_region != (self.normalized_box_xyxy is not None):
            raise ValueError("region-short-phrase records require one normalized box")
        if self.normalized_box_xyxy is not None:
            x0, y0, x1, y1 = self.normalized_box_xyxy
            if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
                raise ValueError("normalized box must satisfy 0<=min<max<=1")
        if self.component == "rsieval" and self.split_policy != "permanent_test_only":
            raise ValueError("RSIEval must remain permanent test-only")
        if self.training_eligible:
            if not self.license.allowed_for_training or self.split_policy == "permanent_test_only":
                raise ValueError("training eligibility requires an approved non-test source")
        return self


__all__ = ["DescriptionSourceRecord", "LanguageAnswer", "LanguageImageRef"]
