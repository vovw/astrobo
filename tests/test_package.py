# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package contracts that run without torch, cuRobo, or a GPU."""

from __future__ import annotations

import json
from types import SimpleNamespace

from astrobo import Astrobo, MotionReport
from astrobo.yam import YamJointLayout


def test_model_context() -> None:
    robot = Astrobo(SimpleNamespace(joint_names=["joint"], tool_frames=["tip"]), frame="base")
    context = robot.context()
    assert context["joint_names"] == ["joint"]
    assert context["quaternion_order"] == "wxyz"
    assert context["execution_supported"] is False
    json.dumps(context, allow_nan=False)


def test_feedback_is_json_ready() -> None:
    report = MotionReport(False, "no_plan", "No accepted path.", "base", 0)
    assert json.loads(json.dumps(report.to_dict()))["trajectory"] is None
    assert report.motion_report() == {"lines": ["astrobo no_plan: No accepted path."]}


def test_yam_layout() -> None:
    layout = YamJointLayout(tuple(f"left_{i}" for i in range(6)),
                            tuple(f"right_{i}" for i in range(6)))
    joints = layout.measured_joints([0.1] * 6 + [1.0] + [0.2] * 6 + [0.0])
    assert len(joints) == 12
    assert joints["left_0"] == 0.1
    assert joints["right_0"] == 0.2
