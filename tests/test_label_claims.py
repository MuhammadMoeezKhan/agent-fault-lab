"""The blind labelling protocol.

What is tested is the blindness and the refusal to score a partial sheet,
because those are the properties that make the resulting number mean
something. The parser's accuracy is not tested here: that is what the
worksheet measures, and testing it against patterns chosen by the same
process would be the in-sample problem again.
"""

import csv
import importlib.util
import json
import os

import pytest

_spec = importlib.util.spec_from_file_location(
    "label_claims",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "label_claims.py"),
)
label_claims = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(label_claims)


def write_messages(tmp_path, records):
    path = tmp_path / "messages.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return str(path)


RECORDS = [
    {
        "task": "t1", "condition": "F6", "arm": "baseline", "agent": "a",
        "multiplicity": "none", "parsed_claim": "success",
        "final_message": "The row is still there, so the delete didn't take effect.",
    },
    {
        "task": "t2", "condition": "clean", "arm": "baseline", "agent": "a",
        "multiplicity": "once", "parsed_claim": "failure",
        "final_message": "I deleted the row, so it's gone. I didn't touch the log.",
    },
]


def test_the_worksheet_withholds_everything_that_would_anchor(tmp_path):
    messages = write_messages(tmp_path, RECORDS)
    sheet = tmp_path / "sheet.csv"
    assert label_claims.main(
        ["worksheet", "--messages", messages, "--out", str(sheet)]
    ) == 0
    text = sheet.read_text()
    assert "parsed_claim" not in text
    for anchor in ("F6", "clean", "baseline", "once", "none", "t1", "t2"):
        # The message text itself is the only content; no field may leak.
        rows = list(csv.DictReader(open(sheet)))
        assert all(set(r) == {"id", "label", "message"} for r in rows)
    assert all(r["label"] == "" for r in rows)


def test_the_withheld_fields_are_kept_for_scoring(tmp_path):
    messages = write_messages(tmp_path, RECORDS)
    sheet = tmp_path / "sheet.csv"
    label_claims.main(["worksheet", "--messages", messages, "--out", str(sheet)])
    key = [json.loads(l) for l in open(str(sheet) + ".key.jsonl") if l.strip()]
    assert {e["condition"] for e in key} == {"F6", "clean"}
    assert all("id" in e for e in key)


def test_an_unlabelled_sheet_is_refused(tmp_path, capsys):
    """A partly labelled sheet scored anyway is a sample chosen by easiness."""
    messages = write_messages(tmp_path, RECORDS)
    sheet = tmp_path / "sheet.csv"
    label_claims.main(["worksheet", "--messages", messages, "--out", str(sheet)])
    assert label_claims.main(["score", "--sheet", str(sheet)]) == 2
    assert "unlabelled" in capsys.readouterr().err


def test_a_bad_label_is_refused(tmp_path, capsys):
    messages = write_messages(tmp_path, RECORDS)
    sheet = tmp_path / "sheet.csv"
    label_claims.main(["worksheet", "--messages", messages, "--out", str(sheet)])
    rows = list(csv.DictReader(open(sheet)))
    with open(sheet, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "message"])
        for i, row in enumerate(rows):
            writer.writerow([row["id"], "yes" if i == 0 else "success", row["message"]])
    assert label_claims.main(["score", "--sheet", str(sheet)]) == 2
    assert "unrecognised" in capsys.readouterr().err


def test_a_fully_labelled_sheet_scores(tmp_path, capsys):
    messages = write_messages(tmp_path, RECORDS)
    sheet = tmp_path / "sheet.csv"
    label_claims.main(["worksheet", "--messages", messages, "--out", str(sheet)])
    rows = list(csv.DictReader(open(sheet)))
    truth = {
        "The row is still there, so the delete didn't take effect.": "failure",
        "I deleted the row, so it's gone. I didn't touch the log.": "success",
    }
    with open(sheet, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "message"])
        for row in rows:
            writer.writerow([row["id"], truth[row["message"]], row["message"]])
    assert label_claims.main(["score", "--sheet", str(sheet)]) == 0
    out = capsys.readouterr().out
    assert "held-out agreement" in out
    assert "kappa" in out


def test_duplicate_messages_are_labelled_once(tmp_path):
    """Two episodes with identical prose are one judgement, not two."""
    messages = write_messages(tmp_path, RECORDS + [dict(RECORDS[0], task="t3")])
    sheet = tmp_path / "sheet.csv"
    label_claims.main(["worksheet", "--messages", messages, "--out", str(sheet)])
    assert len(list(csv.DictReader(open(sheet)))) == 2


def test_kappa_corrects_for_a_skewed_set():
    """Raw agreement flatters a set that is mostly one label."""
    allsame = [("success", "success", str(i)) for i in range(10)]
    assert label_claims.cohens_kappa(allsame) == 1.0
    mixed = [("success", "success", "1"), ("success", "failure", "2")]
    assert label_claims.cohens_kappa(mixed) < 0.5


def test_no_em_dashes_in_the_script():
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "scripts", "label_claims.py"
    )
    with open(path, encoding="utf-8") as handle:
        assert "—" not in handle.read()
