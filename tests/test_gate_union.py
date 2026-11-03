"""The go/no-go gate is a union, and it is derived rather than added up.

This exists because the gate was published once as "16 of 44" after
subtracting three fixtures that both criteria selected, when only one of them
did. The error moved a gate further from GO than the data said, and it was
found only by recomputing. A union cannot be got wrong the same way, and a
test pins it so a later refactor cannot turn it back into a subtraction.
"""

import importlib.util
import json
import pathlib

SPEC = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "combine_reports.py"
_spec = importlib.util.spec_from_file_location("combine_reports", SPEC)
combine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(combine)


def _row(server, fixture, fault, verdict, *, distinguishable, confirmable):
    return {
        "server": server,
        "tool": "write",
        "fixture": fixture,
        "fault": fault,
        "verdict": verdict,
        "mutating": True,
        "deterministic": True,
        "repeat_distinguishable": distinguishable,
        "confirmable": confirmable,
        "multiplicities": ["once"],
        "safety_mechanism": "unknown",
        "idempotent_hint": None,
        "read_only_hint": False,
        "dedup_argument": False,
        "precondition_argument": False,
        "precondition_supplied": False,
        "duplicate_visible": None,
        "caller_left_unsure": False,
        "annotation_violations": [],
        "notes": [],
        "contained": True,
        "image": "test",
    }


def _write(tmp_path, rows):
    path = tmp_path / "audit-test.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return tmp_path


def _pool(tmp_path, rows):
    directory = _write(tmp_path, rows)
    return combine.describe(combine.load(str(directory)))


def test_a_fixture_selected_by_both_criteria_is_counted_once(tmp_path):
    """The exact shape of the published error: an overlapping fixture."""
    rows = [
        # Duplicates and is unconfirmable: in both sets, one fixture.
        _row("s1", "both", "F1", "DUPLICATED", distinguishable=True, confirmable=False),
        # Duplicates only.
        _row("s2", "dup", "F1", "DUPLICATED", distinguishable=True, confirmable=True),
        # Unconfirmable only.
        _row("s3", "unc", "F1", "SAFE", distinguishable=True, confirmable=False),
    ]
    pooled = _pool(tmp_path, rows)
    assert pooled["mutating"] == 3
    assert pooled["unsafe_or_unconfirmable"].startswith("100%")
    assert pooled["unsafe_or_unconfirmable_overlap"] == 1


def test_the_union_is_not_the_sum(tmp_path):
    """Two rows, both criteria, one fixture: the sum would say two."""
    rows = [
        _row("s1", "both", "F1", "DUPLICATED", distinguishable=True, confirmable=False),
        _row("s2", "safe", "F1", "SAFE", distinguishable=True, confirmable=True),
    ]
    pooled = _pool(tmp_path, rows)
    assert pooled["unsafe_or_unconfirmable"].startswith("50%")
    assert pooled["unsafe_or_unconfirmable_overlap"] == 1


def test_an_unknown_confirmability_is_not_counted_as_unconfirmable(tmp_path):
    """`None` means the read path could not answer, which is not a finding."""
    rows = [
        _row("s1", "unknown", "F1", "SAFE", distinguishable=True, confirmable=None),
    ]
    pooled = _pool(tmp_path, rows)
    assert pooled["unsafe_or_unconfirmable"].startswith("0%")
    assert pooled["unsafe_or_unconfirmable_overlap"] == 0


def test_an_unproven_fixture_is_in_no_denominator(tmp_path):
    rows = [
        _row("s1", "bad", "F1", "UNPROVEN", distinguishable=True, confirmable=False),
        _row("s2", "dup", "F1", "DUPLICATED", distinguishable=True, confirmable=True),
    ]
    pooled = _pool(tmp_path, rows)
    assert pooled["mutating"] == 1
    assert pooled["unsafe_or_unconfirmable"].startswith("100%")
