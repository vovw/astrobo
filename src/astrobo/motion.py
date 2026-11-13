# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-request IK and planning tools; no hardware execution."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from threading import RLock
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from curobo.motion_planner import MotionPlanner
    from curobo.scene import Scene
    from curobo.types import GoalToolPose, JointState


def _invalid(message: str) -> None:
    """Raise an input error through cuRobo's logging helper."""
    from curobo._src.util.logging import log_and_raise

    log_and_raise(message, logger_name="astrobo", exc_info=False)


@dataclass(frozen=True)
class MotionReport:
    """JSON-ready result, with no trajectory on failure."""

    success: bool  # Whether the configured solver accepted the request.
    status: str  # solved, not_solved, planned, or no_plan.
    message: str  # Grounded feedback without an inferred failure cause.
    frame: str  # Configured robot base/planning frame.
    scene_revision: int  # Revision of scenes updated through this wrapper.
    trajectory: dict[str, Any] | None = None  # Trimmed joint trajectory; never motor commands.

    def to_dict(self) -> dict[str, Any]:
        """Return a detached dictionary for an agent tool response."""
        return asdict(self)

    def motion_report(self) -> dict[str, list[str]]:
        """Format feedback for Inspect Robots observation.extra['motion_report']."""
        return {"lines": [f"astrobo {self.status}: {self.message}"]}


class Astrobo:
    """Wrap one caller-owned, warmed MotionPlanner and its existing IK solver.

    Goals contain every configured tool, each as xyz + wxyz in the planning frame.
    Joint mappings describe a freshly measured stationary start. This v0.1 interface
    does not support moving starts, execution, perception, or coordinate transforms.
    """

    def __init__(self, planner: MotionPlanner, *, frame: str) -> None:
        if not frame.strip():
            _invalid("frame must name the configured robot base frame")
        self._planner = planner
        self.frame = frame
        self._revision = 0
        self._lock = RLock()

    def context(self) -> dict[str, Any]:
        """Describe the model contract without inventing observations."""
        return {
            "frame": self.frame,
            "joint_names": list(self._planner.joint_names),
            "tool_frames": list(self._planner.tool_frames),
            "units": {"position": "m", "joint": "rad", "time": "s"},
            "quaternion_order": "wxyz",
            "scene_revision": self._revision,
            "all_tools_required": True,
            "stationary_start_required": True,
            "execution_supported": False,
        }

    def _inputs(
        self, joints: Mapping[str, float], goals: Mapping[str, Sequence[float]],
    ) -> tuple[JointState, GoalToolPose]:
        """Validate names and finite values, then allocate float32 CUDA inputs."""
        import torch
        from curobo.types import GoalToolPose, JointState, Pose

        names = list(self._planner.joint_names)
        if set(joints) != set(names):
            _invalid(f"joints must contain exactly these measured joints: {names}")
        if set(goals) != set(self._planner.tool_frames):
            _invalid(f"goals must contain every configured tool: {self._planner.tool_frames}")
        positions = [float(joints[name]) for name in names]
        if not all(math.isfinite(value) for value in positions):
            _invalid("joint positions must be finite")
        normalized = {}
        for tool, pose in goals.items():
            values = [float(value) for value in pose]
            if len(values) != 7 or not all(math.isfinite(value) for value in values):
                _invalid(f"{tool}: expected seven finite values: xyz + wxyz")
            norm = math.hypot(*values[3:])
            if norm < 1e-8:
                _invalid(f"{tool}: quaternion must be nonzero")
            normalized[tool] = values[:3] + [value / norm for value in values[3:]]
        device = self._planner.default_joint_state.position.device
        if device.type != "cuda":
            _invalid("astrobo requires a CUDA planner")
        state = JointState.from_position(
            torch.tensor([positions], device=device, dtype=torch.float32), joint_names=names,
        )
        poses = {
            name: Pose(
                position=torch.tensor([pose[:3]], device=device, dtype=torch.float32),
                quaternion=torch.tensor([pose[3:]], device=device, dtype=torch.float32),
            )
            for name, pose in normalized.items()
        }
        return state, GoalToolPose.from_poses(poses)

    def check_reachability(
        self, joints: Mapping[str, float], goals: Mapping[str, Sequence[float]],
    ) -> MotionReport:
        """Check one simultaneous tool-pose request; failure is not proof of unreachability."""
        with self._lock:
            state, goal = self._inputs(joints, goals)
            result = self._planner.ik_solver.solve_pose(goal, current_state=state)
            success = _single_success(result)
            return MotionReport(
                success, "solved" if success else "not_solved",
                "IK found a configuration. The route has not been checked."
                if success else "IK found no accepted solution within its configured budget.",
                self.frame, self._revision,
            )

    def plan_motion(
        self, joints: Mapping[str, float], goals: Mapping[str, Sequence[float]],
        *, max_attempts: int = 2,
    ) -> MotionReport:
        """Return a trimmed trajectory after single-problem success, or a failure report."""
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            _invalid("max_attempts must be a positive integer")
        with self._lock:
            state, goal = self._inputs(joints, goals)
            result = self._planner.plan_pose(goal, state, max_attempts=max_attempts)
            if not _single_success(result):
                return MotionReport(
                    False, "no_plan", "No accepted path found within the configured budget.",
                    self.frame, self._revision,
                )
            trajectory = _trajectory(result.get_interpolated_plan())
            return MotionReport(
                True, "planned", "Planner accepted this path against its configured model and scene. "
                "This is a preview, not execution authorization.",
                self.frame, self._revision, trajectory,
            )

    def update_world(self, scene: Scene) -> None:
        """Replace the full scene within existing capacities, serialized against solves."""
        with self._lock:
            self._planner.update_world(scene)
            self._revision += 1


def _single_success(result: Any) -> bool:
    """Reject unsupported batch results rather than accepting any successful row."""
    if result is None:
        return False
    if result.success.numel() != 1:
        _invalid("astrobo expects exactly one problem and one returned solution")
    return bool(result.success.item())


def _trajectory(state: JointState) -> dict[str, Any]:
    """Serialize a trimmed single trajectory with its original joint order and timing."""
    import torch

    position = state.position
    while position.ndim > 2 and position.shape[0] == 1:
        position = position.squeeze(0)
    names = state.joint_names
    if position.ndim != 2 or position.shape[0] < 1 or names is None:
        _invalid("planner returned an invalid trajectory shape or missing joint names")
    if len(names) != position.shape[-1] or len(set(names)) != len(names):
        _invalid("trajectory joint names do not match its columns")
    if not bool(torch.isfinite(position).all()):
        _invalid("planner returned non-finite trajectory positions")
    if state.dt is None or state.dt.numel() != 1:
        _invalid("trajectory must contain one timestep duration")
    dt = float(state.dt.item())
    if not math.isfinite(dt) or dt <= 0:
        _invalid("trajectory timestep must be finite and positive")
    output = {
        "joint_names": list(names), "position": position.detach().cpu().tolist(),
        "dt": dt, "duration": (position.shape[0] - 1) * dt,
    }
    for field in ("velocity", "acceleration", "jerk"):
        value = getattr(state, field, None)
        if value is not None:
            if value.shape != state.position.shape or not bool(torch.isfinite(value).all()):
                _invalid(f"invalid trajectory {field}")
            output[field] = value.reshape(position.shape).detach().cpu().tolist()
    return output
