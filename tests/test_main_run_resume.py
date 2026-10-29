"""Tests for planning, resuming and recording the main run.

The main run is 288 episodes an arm and the hosted arm is capped at 35 a
night, so it spans about nine nights and will be stopped part way by the cap,
by a client refusal and probably by the machine. Three properties have to hold
for that to be safe, and none of them can be checked by running it:

- a run stopped half way holds whole task clusters, because the primary test
  is a sign test over clusters;
- resuming spends no episode on a cell already on disk, including one that
  errored;
- rows reach the disk as episodes finish, and a resumed batch appends rather
  than truncating what the earlier one paid for.
"""
from __future__ import annotations

import csv
import os

from agentfaultlab.consequence import (
    AgentTask,
    Cell,
    EpisodeResult,
    RowWriter,
    plan_cells,
    recorded_cells,
    remaining_cells,
)

CONDITIONS = ("clean", "F1", "F6")
ARMS = ("baseline", "unconfirmed_allowed")


def _task(name):
    return AgentTask(id=name, instruction="do it", reference=[], target_tool="write")


def _result(task, condition, arm, error=None):
    return EpisodeResult(
        task=task,
        agent="test-agent",
        condition=condition,
        arm=arm,
        multiplicity="once",
        claim="success",
        task_success=True,
        duplicate_effect=False,
        false_success=False,
        false_failure=False,
        honest_unconfirmed=False,
        error=error,
    )


def _write_rows(path, rows):
    writer = RowWriter(path)
    for row in rows:
        writer.write(row)
    writer.close()
    return path


def test_the_plan_finishes_a_cluster_before_starting_the_next():
    cells = plan_cells([_task("a"), _task("b")], CONDITIONS, ARMS, 3)
    assert len(cells) == 2 * 3 * 2 * 3
    # Every cell of task "a" comes before any cell of task "b". A run cut off
    # by the cap then holds whole clusters, which the sign test can use, rather
    # than one condition of every cluster, which it cannot.
    tasks = [cell.task for cell in cells]
    assert tasks == ["a"] * 18 + ["b"] * 18
    assert cells[0] == Cell("a", "clean", "baseline", 1)


def test_resuming_skips_what_is_already_recorded(tmp_path):
    path = _write_rows(
        str(tmp_path / "run.csv"),
        [_result("a", "clean", "baseline") for _ in range(2)],
    )
    grid = plan_cells([_task("a")], ("clean",), ("baseline",), 3)
    left = remaining_cells(grid, recorded_cells([path]))
    assert len(left) == 1
    assert left[0].task == "a"


def test_an_errored_episode_counts_as_run(tmp_path):
    # Tempting to re-run, since it contributes to no denominator. But a client
    # or server that fails reproducibly would then be retried every night out
    # of a cap that cannot be given back.
    path = _write_rows(
        str(tmp_path / "run.csv"), [_result("a", "F1", "baseline", error="boom")]
    )
    grid = plan_cells([_task("a")], ("F1",), ("baseline",), 1)
    assert remaining_cells(grid, recorded_cells([path])) == []


def test_cells_are_matched_by_count_not_by_repeat_index(tmp_path):
    # Nothing distinguishes the second episode of a cell from the third, so a
    # file whose rows arrived in another order must not cause a re-run.
    path = _write_rows(
        str(tmp_path / "run.csv"),
        [_result("a", "clean", "baseline"), _result("a", "F6", "baseline")],
    )
    grid = plan_cells([_task("a")], ("clean", "F6"), ("baseline",), 1)
    assert remaining_cells(grid, recorded_cells([path])) == []


def test_a_different_arm_is_a_different_cell(tmp_path):
    path = _write_rows(
        str(tmp_path / "run.csv"), [_result("a", "clean", "baseline")]
    )
    grid = plan_cells([_task("a")], ("clean",), ARMS, 1)
    left = remaining_cells(grid, recorded_cells([path]))
    assert [cell.arm for cell in left] == ["unconfirmed_allowed"]


def test_rows_are_readable_after_each_episode(tmp_path):
    path = str(tmp_path / "run.csv")
    writer = RowWriter(path)
    writer.write(_result("a", "clean", "baseline"))
    # Not closed: an interrupted run must still leave the row behind.
    with open(path, newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1
    writer.write(_result("a", "F1", "baseline"))
    with open(path, newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    writer.close()


def test_a_second_batch_appends_and_writes_one_header(tmp_path):
    path = str(tmp_path / "run.csv")
    _write_rows(path, [_result("a", "clean", "baseline")])
    _write_rows(path, [_result("b", "clean", "baseline")])
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["task"] for row in rows] == ["a", "b"]
    with open(path) as handle:
        assert handle.read().count("multiplicity") == 1


def test_a_missing_resume_file_is_not_an_error(tmp_path):
    # The first night of a run passes --resume on a file that does not exist
    # yet, and that must plan the whole grid rather than refuse.
    grid = plan_cells([_task("a")], ("clean",), ("baseline",), 1)
    missing = str(tmp_path / "nope.csv")
    assert remaining_cells(grid, recorded_cells([missing])) == grid


def test_the_test_file_writes_no_em_dashes():
    with open(__file__) as handle:
        assert chr(0x2014) not in handle.read()
