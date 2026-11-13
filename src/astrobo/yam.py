# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit mapping from Inspect Robots YAM observations to named arm joints."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from astrobo.motion import _invalid


@dataclass(frozen=True)
class YamJointLayout:
    """Map the left-then-right 14-slot observation; grippers are not arm angles."""

    left: tuple[str, ...]  # Six model joint names, in left hardware observation order.
    right: tuple[str, ...]  # Six model joint names, in right hardware observation order.

    def __post_init__(self) -> None:
        names = (*self.left, *self.right)
        if len(self.left) != 6 or len(self.right) != 6 or len(set(names)) != 12:
            _invalid("YAM requires six unique joint names per arm, twelve in total")
        if not all(isinstance(name, str) and name.strip() for name in names):
            _invalid("YAM joint names must be nonempty strings")

    def measured_joints(self, joint_pos: Sequence[float]) -> dict[str, float]:
        """Decode [left six radians, aperture, right six radians, aperture]."""
        values = [float(value) for value in joint_pos]
        if len(values) != 14 or not all(math.isfinite(value) for value in values):
            _invalid("YAM joint_pos must contain fourteen finite values")
        if not all(0.0 <= values[index] <= 1.0 for index in (6, 13)):
            _invalid("YAM gripper apertures must be normalized to [0, 1]")
        return dict(zip((*self.left, *self.right), values[:6] + values[7:13]))
