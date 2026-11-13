# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract checks for the standalone astrobo package."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")

from astrobo import Astrobo, MotionReport
from astrobo.motion import _single_success, _trajectory
from astrobo.yam import YamJointLayout


def test_yam_mapping_excludes_grippers() -> None:
    layout = YamJointLayout(tuple(f"l{i}" for i in range(6)), tuple(f"r{i}" for i in range(6)))
    result = layout.measured_joints([0.1] * 6 + [1.0] + [0.2] * 6 + [0.0])
    assert result == {**{f"l{i}": 0.1 for i in range(6)}, **{f"r{i}": 0.2 for i in range(6)}}
    with pytest.raises(ValueError):
        layout.measured_joints([float("nan")] * 14)
    with pytest.raises(ValueError):
        layout.measured_joints([2.0] * 14)


def test_single_success_rejects_partial_batch() -> None:
    assert not _single_success(None)
    with pytest.raises(ValueError):
        _single_success(SimpleNamespace(success=torch.tensor([True, False])))


def test_failed_plan_never_exposes_trajectory(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = Mock()
    failed = Mock(success=torch.tensor([False]))
    planner.plan_pose.return_value = failed
    robot = Astrobo(planner, frame="base")
    monkeypatch.setattr(robot, "_inputs", lambda *args: ("measured", "goal"))
    report = robot.plan_motion({}, {})
    assert not report.success and report.trajectory is None
    failed.get_interpolated_plan.assert_not_called()
    json.dumps(report.to_dict(), allow_nan=False)


def test_reuses_owned_ik(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = Mock()
    planner.ik_solver.solve_pose.return_value = SimpleNamespace(success=torch.tensor([True]))
    robot = Astrobo(planner, frame="base")
    monkeypatch.setattr(robot, "_inputs", lambda *args: ("measured", "goal"))
    assert robot.check_reachability({}, {}).success
    planner.ik_solver.solve_pose.assert_called_once_with("goal", current_state="measured")


def test_plan_uses_trimmed_output(monkeypatch: pytest.MonkeyPatch) -> None:
    planner = Mock()
    result = Mock(success=torch.tensor([True]))
    result.get_interpolated_plan.return_value = SimpleNamespace(
        position=torch.tensor([[0.0], [0.2]], dtype=torch.float32),
        joint_names=["a"], dt=torch.tensor(0.05, dtype=torch.float32),
    )
    planner.plan_pose.return_value = result
    robot = Astrobo(planner, frame="base")
    monkeypatch.setattr(robot, "_inputs", lambda *args: ("measured", "goal"))
    report = robot.plan_motion({}, {}, max_attempts=3)
    assert report.success
    assert len(report.trajectory["position"]) == 2
    result.get_interpolated_plan.assert_called_once_with()
    planner.plan_pose.assert_called_once_with("goal", "measured", max_attempts=3)
    json.dumps(report.to_dict(), allow_nan=False)


def test_trimmed_trajectory_preserves_order_and_time() -> None:
    state = SimpleNamespace(
        position=torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32),
        joint_names=["right", "left"], dt=torch.tensor(0.1, dtype=torch.float32),
    )
    result = _trajectory(state)
    assert result["position"] == [[1.0, 2.0], [3.0, 4.0]]
    assert result["joint_names"] == ["right", "left"]
    assert result["duration"] == pytest.approx(0.1)
    state.dt = torch.tensor(0.0, dtype=torch.float32)
    with pytest.raises(ValueError):
        _trajectory(state)


def test_report_adapter_and_copy() -> None:
    report = MotionReport(True, "planned", "Accepted.", "base", 2, {"position": [[1.0]]})
    serialized = report.to_dict()
    serialized["trajectory"]["position"][0][0] = 7.0
    assert report.trajectory["position"] == [[1.0]]
    assert report.motion_report() == {"lines": ["astrobo planned: Accepted."]}


def test_scene_revision_changes_only_after_success() -> None:
    planner = Mock(joint_names=["a"], tool_frames=["tip"])
    robot = Astrobo(planner, frame="base")
    robot.update_world("scene")
    assert robot.context()["scene_revision"] == 1
    planner.update_world.side_effect = RuntimeError("capacity exceeded")
    with pytest.raises(RuntimeError):
        robot.update_world("too large")
    assert robot.context()["scene_revision"] == 1


@pytest.mark.parametrize("joints,goals", [
    ({"wrong": 0.0}, {"tip": [0, 0, 0, 1, 0, 0, 0]}),
    ({"a": float("nan")}, {"tip": [0, 0, 0, 1, 0, 0, 0]}),
    ({"a": 0.0}, {"tip": [0, 0, 0, 0, 0, 0, 0]}),
    ({"a": 0.0}, {}),
])
def test_bad_inputs_fail_before_solver(joints: dict, goals: dict) -> None:
    planner = Mock(joint_names=["a"], tool_frames=["tip"])
    robot = Astrobo(planner, frame="base")
    with pytest.raises(ValueError):
        robot.check_reachability(joints, goals)
    planner.ik_solver.solve_pose.assert_not_called()
