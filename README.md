<div align="center">

# astrobo

**Grounded motion tools for robot agents.**

cuRobo planning · YAM joint mapping · Inspect Robots feedback

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Status](https://img.shields.io/badge/status-experimental-orange)

</div>

Give an agent a named pose request. Get a feasibility result or a trajectory preview
back—with explicit units, joint ordering, and feedback it can reason about.

```python
from astrobo import Astrobo

# Reuse your configured, warmed cuRobo MotionPlanner.
robot = Astrobo(planner, frame="base")

context = robot.context()
result = robot.plan_motion(measured_joints, goals)
feedback = result.to_dict()
```

`measured_joints` maps model joint names to radians. `goals` maps every configured
tool frame to `[x, y, z, qw, qx, qy, qz]` in the robot's planning frame.

### Small on purpose

- **Check reachability** using the planner's existing IK solver.
- **Preview motion** with trimmed trajectories and original timing.
- **Give agents grounded feedback** without inventing reasons for solver failure.
- **Decode YAM observations** with explicit joint names and separate gripper apertures.
- **Format Inspect Robots feedback** through its `motion_report` convention.

### Install

Inside an environment with a compatible cuRobo installation:

```bash
python -m pip install -e .
```

GPU dependencies are installed separately. This package targets the public
`MotionPlanner` API in the cuRobo checkout identified in the
[usage guide](docs/usage.md); older cuRobo `MotionGen` releases are not compatible.

### Scope

**v0.1 is a planning library, not a hardware controller.** It assumes freshly
measured stationary starts. It does not include a calibrated YAM model,
perception, execution, or an automatically registered Inspect Robots policy.

IK success checks a configuration, not the route. Planning results depend on the
robot model and scene you supply. GPU and real-YAM validation remain outstanding.

[Usage & integration](docs/usage.md) · [Contributing](CONTRIBUTING.md) · [License](LICENSE)
