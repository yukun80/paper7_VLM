from __future__ import annotations

import gc
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from tests.data.grounded.fixture_helpers import no_target_output, target_output
from tests.data.grounded.single_expert_fixture_helpers import build_annotation_asset, draft_run

from oa_groundrag.data.grounded.annotation.project import (
    AnnotationIntendedUse,
    annotation_work_item,
    create_annotation_project,
    load_annotation_project,
    load_verified_work,
    write_draft_results,
)
from oa_groundrag.data.grounded.annotation.workbench import (
    ANNOTATION_FORM_FIELDS,
    annotation_view_item,
    apply_advanced_json,
    apply_annotation_action,
    create_annotation_app,
    description_to_form_values,
    ensure_loopback_proxy_bypass,
    form_interactive_flags,
    form_values_to_description,
    preview_form_values,
    serve_annotation_workbench,
)
from oa_groundrag.data.grounded.contracts import LandslideEvidenceError
from oa_groundrag.data.rs_general.io import canonical_json, sha256_text
from oa_groundrag.grounding.outputs import region_output_template


REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT = REPO_ROOT / "configs/grounding/prompts/single_expert_prompt_v1.txt"


class SingleExpertWorkbenchTest(unittest.TestCase):
    def _project(
        self,
        root: Path,
        *,
        invalid: bool = False,
        target_status: str | None = None,
    ) -> tuple[Path, dict]:
        asset_root = build_annotation_asset(root / "asset", kind="train")
        project_root = root / "project"
        create_annotation_project(
            asset_root=asset_root,
            output_root=project_root,
            intended_use=AnnotationIntendedUse.TRAIN_SUPERVISION.value,
            prompt_path=PROMPT,
        )
        _, assignments = load_annotation_project(project_root)
        assignment = next(
            row
            for row in assignments
            if row["partition"] == "calibration"
            and (target_status is None or row["target_status"] == target_status)
        )
        description = no_target_output() if assignment["target_status"] == "no_target" else target_output()
        run = draft_run(
            [assignment["record_id"]],
            partition=assignment["partition"],
            suffix="ui",
            prompt_text=(project_root / "prompt.txt").read_text(encoding="utf-8"),
        )
        draft = {
            "schema_version": "oa_groundrag.mask_grounded_region.model_draft.v1",
            "draft_id": "draft-ui",
            "draft_run_id": run["draft_run_id"],
            "record_id": assignment["record_id"],
            "asset_identity_sha256": assignment["asset_identity_sha256"],
            "messages_sha256": sha256_text("ui-messages"),
            "raw_output": "not-json" if invalid else canonical_json(description),
            "parse_status": "invalid" if invalid else "valid",
            "description": None if invalid else description,
            "failure": {
                "schema_version": "oa_groundrag.mask_grounded_region.model_draft_failure.v1",
                "code": "INVALID_MODEL_OUTPUT",
                "message": "fixture",
                "details": {},
            } if invalid else None,
        }
        write_draft_results(
            project_root,
            draft_run=run,
            new_drafts=[draft],
            freeze_prompt=False,
        )
        return project_root, assignment

    def test_save_restore_and_verify_callbacks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root, assignment = self._project(Path(temporary))
            edited = no_target_output() if assignment["target_status"] == "no_target" else target_output()
            edited["short_summary"] = "专家核验后的当前影像描述。"
            editor_text = json.dumps(edited, ensure_ascii=False, indent=2)
            saved = apply_annotation_action(
                project_root=project_root,
                action="save",
                ordinal=0,
                record_id=assignment["record_id"],
                editor_text=editor_text,
                partition="calibration",
                view="pending",
            )
            self.assertIn("已原子保存", saved[-1])
            snapshot = json.loads(
                (
                    project_root
                    / "work"
                    / f"{assignment['record_id']}.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(snapshot["annotator"], "expert")
            self.assertNotIn("annotator_id", snapshot)
            self.assertEqual(
                annotation_work_item(project_root, assignment["ordinal"])["editor_text"],
                editor_text,
            )
            verified = apply_annotation_action(
                project_root=project_root,
                action="verify",
                ordinal=0,
                record_id=assignment["record_id"],
                editor_text=editor_text,
                partition="calibration",
                view="pending",
            )
            self.assertIn("expert_verified", verified[-1])
            self.assertEqual(verified[2], "")
            self.assertIn("没有可显示记录", verified[-1])
            self.assertIn(assignment["record_id"], load_verified_work(project_root))

            reopened = annotation_view_item(
                project_root,
                0,
                partition="calibration",
                view="all",
            )
            self.assertEqual(reopened["record_id"], assignment["record_id"])
            self.assertTrue(reopened["verified"])

            revised = dict(edited)
            revised["short_summary"] = "专家重新打开后保存、尚待再次核验的修改。"
            revised_text = json.dumps(revised, ensure_ascii=False, indent=2)
            apply_annotation_action(
                project_root=project_root,
                action="save",
                ordinal=0,
                record_id=assignment["record_id"],
                editor_text=revised_text,
                partition="calibration",
                view="all",
            )
            self.assertEqual(
                annotation_view_item(
                    project_root,
                    0,
                    partition="calibration",
                    view="all",
                )["editor_text"],
                revised_text,
            )

    def test_invalid_model_draft_loads_strict_empty_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root, assignment = self._project(Path(temporary), invalid=True)
            item = annotation_view_item(
                project_root,
                0,
                partition="calibration",
                view="pending",
            )
            editor = json.loads(item["editor_text"])
            self.assertEqual(editor["target_status"], assignment["target_status"])
            self.assertEqual(item["status"]["failed_drafts"], 1)

    def test_low_information_can_be_saved_but_cannot_be_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project_root, assignment = self._project(
                Path(temporary),
                target_status="target_present",
            )
            editor_text = json.dumps(
                region_output_template("target_present"),
                ensure_ascii=False,
            )
            saved = apply_annotation_action(
                project_root=project_root,
                action="save",
                ordinal=0,
                record_id=assignment["record_id"],
                editor_text=editor_text,
                partition="calibration",
                view="pending",
            )
            self.assertIn("已原子保存", saved[-1])
            with self.assertRaises(LandslideEvidenceError) as raised:
                apply_annotation_action(
                    project_root=project_root,
                    action="verify",
                    ordinal=0,
                    record_id=assignment["record_id"],
                    editor_text=editor_text,
                    partition="calibration",
                    view="pending",
                )
            self.assertEqual(raised.exception.code, "ANNOTATION_LOW_INFORMATION")

    def test_form_canonical_and_advanced_json_are_bidirectionally_consistent(self) -> None:
        description = target_output()
        values = description_to_form_values(description)
        self.assertEqual(len(values), len(ANNOTATION_FORM_FIELDS))
        self.assertEqual(form_values_to_description(values), description)
        preview, editor_text, message = preview_form_values(values)
        self.assertEqual(preview, description)
        self.assertEqual(json.loads(editor_text), description)
        self.assertIn("严格", message)
        restored, restored_preview, normalized = apply_advanced_json(
            editor_text,
            expected_target_status="target_present",
        )
        self.assertEqual(restored, values)
        self.assertEqual(restored_preview, description)
        self.assertEqual(json.loads(normalized), description)

        duplicated = list(values)
        duplicated[9] = "vegetation\nvegetation"
        with self.assertRaises(Exception):
            form_values_to_description(duplicated)
        with self.assertRaises(Exception):
            apply_advanced_json(
                json.dumps(no_target_output()),
                expected_target_status="target_present",
            )

    def test_target_and_no_target_form_lock_contract(self) -> None:
        target_flags = form_interactive_flags("target_present")
        no_target_flags = form_interactive_flags("no_target")
        self.assertFalse(target_flags[0])
        self.assertTrue(all(target_flags[1:]))
        self.assertFalse(no_target_flags[0])
        self.assertTrue(all(not no_target_flags[index] for index in range(1, 19)))
        self.assertTrue(all(no_target_flags[index] for index in range(19, 22)))
        self.assertFalse(any(form_interactive_flags("", has_record=False)))

    def test_app_has_visible_canonical_json_and_advanced_textbox_not_code(self) -> None:
        with warnings.catch_warnings(), tempfile.TemporaryDirectory() as temporary:
            warnings.simplefilter("ignore", ResourceWarning)
            project_root, _ = self._project(Path(temporary))
            app = create_annotation_app(
                project_root=project_root,
                partition="calibration",
                view="pending",
            )
            config = app.get_config_file()
            app.close()
            del app
            gc.collect()
        components = config["components"]
        by_label = {
            component.get("props", {}).get("label"): component
            for component in components
            if component.get("props", {}).get("label")
        }
        canonical = by_label["专家最终 canonical JSON（只读，始终可见）"]
        advanced = by_label["高级 JSON；修改后点击“应用 JSON 到表单”"]
        self.assertEqual(canonical["type"], "json")
        self.assertEqual(advanced["type"], "textbox")
        self.assertFalse(any(component["type"] == "code" for component in components))

    def test_sparse_partition_filter_and_pending_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_root = build_annotation_asset(root / "asset", kind="train")
            project_root = root / "project"
            create_annotation_project(
                asset_root=asset_root,
                output_root=project_root,
                intended_use=AnnotationIntendedUse.TRAIN_SUPERVISION.value,
                prompt_path=PROMPT,
            )
            _, assignments = load_annotation_project(project_root)
            calibration = [row for row in assignments if row["partition"] == "calibration"]
            remaining = [row for row in assignments if row["partition"] == "remaining"]
            self.assertEqual(len(calibration), 20)
            self.assertTrue(any(
                right["ordinal"] - left["ordinal"] > 1
                for left, right in zip(calibration, calibration[1:])
            ))

            def add_drafts(rows: list[dict], partition: str) -> None:
                run = draft_run(
                    [row["record_id"] for row in rows],
                    partition=partition,
                    suffix=f"view-{partition}",
                    prompt_text=(project_root / "prompt.txt").read_text(encoding="utf-8"),
                )
                drafts = []
                for index, assignment in enumerate(rows):
                    description = (
                        no_target_output()
                        if assignment["target_status"] == "no_target"
                        else target_output()
                    )
                    drafts.append({
                        "schema_version": "oa_groundrag.mask_grounded_region.model_draft.v1",
                        "draft_id": f"draft-view-{partition}-{index}",
                        "draft_run_id": run["draft_run_id"],
                        "record_id": assignment["record_id"],
                        "asset_identity_sha256": assignment["asset_identity_sha256"],
                        "messages_sha256": sha256_text(f"view-{partition}-{index}"),
                        "raw_output": canonical_json(description),
                        "parse_status": "valid",
                        "description": description,
                        "failure": None,
                    })
                write_draft_results(
                    project_root,
                    draft_run=run,
                    new_drafts=drafts,
                    freeze_prompt=False,
                )

            add_drafts([calibration[0], calibration[-1]], "calibration")
            add_drafts([remaining[0]], "remaining")
            first = annotation_view_item(
                project_root,
                0,
                partition="calibration",
                view="pending",
            )
            second = annotation_view_item(
                project_root,
                1,
                partition="calibration",
                view="pending",
            )
            self.assertEqual(first["record_id"], calibration[0]["record_id"])
            self.assertEqual(second["record_id"], calibration[-1]["record_id"])
            self.assertEqual(first["total"], 2)
            self.assertNotEqual(second["global_ordinal"], second["ordinal"])

            editor_text = first["editor_text"]
            advanced = apply_annotation_action(
                project_root=project_root,
                action="verify",
                ordinal=0,
                record_id=first["record_id"],
                editor_text=editor_text,
                partition="calibration",
                view="pending",
            )
            self.assertEqual(advanced[2], calibration[-1]["record_id"])

    def test_loopback_proxy_bypass_keeps_wildcard_and_adds_exact_hosts(self) -> None:
        environ = {
            "NO_PROXY": "example.org,127.*",
            "no_proxy": "internal.example,127.*",
        }
        updated = ensure_loopback_proxy_bypass(environ)
        self.assertEqual(
            updated["NO_PROXY"],
            "example.org,127.*,127.0.0.1,localhost",
        )
        self.assertEqual(
            updated["no_proxy"],
            "internal.example,127.*,127.0.0.1,localhost",
        )

    def test_loopback_proxy_bypass_preserves_order_deduplicates_and_is_idempotent(self) -> None:
        environ = {
            "NO_PROXY": "example.org, localhost,example.org,127.0.0.1",
            "no_proxy": "localhost,service.local,localhost",
        }
        first = ensure_loopback_proxy_bypass(environ)
        second = ensure_loopback_proxy_bypass(environ)
        self.assertEqual(first, second)
        self.assertEqual(
            environ["NO_PROXY"],
            "example.org,localhost,127.0.0.1",
        )
        self.assertEqual(
            environ["no_proxy"],
            "localhost,service.local,127.0.0.1",
        )

    def test_launch_failure_closes_partial_app_and_uses_structured_error(self) -> None:
        class FailingApp:
            def __init__(self) -> None:
                self.launch_options: dict | None = None
                self.close_calls = 0

            def launch(self, **options: object) -> None:
                self.launch_options = dict(options)
                raise RuntimeError("startup-events returned 503")

            def close(self) -> None:
                self.close_calls += 1

        with tempfile.TemporaryDirectory() as temporary:
            project_root, _ = self._project(Path(temporary))
            app = FailingApp()
            with (
                patch(
                    "oa_groundrag.data.grounded.annotation.workbench.create_annotation_app",
                    return_value=app,
                ),
                patch.dict(
                    os.environ,
                    {"NO_PROXY": "127.*", "no_proxy": "proxy.example"},
                    clear=False,
                ),
            ):
                with self.assertRaises(LandslideEvidenceError) as raised:
                    serve_annotation_workbench(
                        project_root=project_root,
                        partition="calibration",
                        auto_close_partition=True,
                    )
                no_proxy_values = {
                    "NO_PROXY": os.environ["NO_PROXY"],
                    "no_proxy": os.environ["no_proxy"],
                }
            self.assertEqual(raised.exception.code, "ANNOTATION_UI_START_FAILED")
            self.assertEqual(
                raised.exception.details,
                {
                    "error_type": "RuntimeError",
                    "error_message": "startup-events returned 503",
                },
            )
            self.assertEqual(app.close_calls, 1)
            self.assertIsNotNone(app.launch_options)
            self.assertTrue(app.launch_options["prevent_thread_lock"])
            self.assertEqual(
                no_proxy_values["NO_PROXY"],
                "127.*,127.0.0.1,localhost",
            )
            self.assertEqual(
                no_proxy_values["no_proxy"],
                "proxy.example,127.0.0.1,localhost",
            )


if __name__ == "__main__":
    unittest.main()
