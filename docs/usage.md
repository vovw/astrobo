# Use astrobo with cuRobo

Small, grounded motion tools for robot agents. Bring a qualified cuRobo robot model
and a warmed planner; get IK feasibility, trajectory previews, and structured feedback.
Version 0.1 does not command hardware or register an Inspect Robots policy.

## Use with your planner

Install into the environment containing this cuRobo checkout and its CUDA dependencies:

```bash
python -m pip install -e .
```

Wrap the planner once after its normal construction and warmup:

```python
from astrobo import Astrobo

robot = Astrobo(planner, frame="base")  # Use the actual configured robot base frame.
context = robot.context()

# Obtain a fresh named joint mapping from the stationary robot, and calibrated goals.
# Every configured tool must have a goal (including the arm you intend to hold).
feasibility = robot.check_reachability(measured_joints, goals)
preview = robot.plan_motion(measured_joints, goals)
tool_response = preview.to_dict()
```

Goals map model tool names to `[x, y, z, qw, qx, qy, qz]`. Coordinates use meters in
the configured robot base frame; joints use radians. Quaternions are normalized.
The wrapper assumes a stationary start: do not use it while the arms are moving.
The planner retains ownership of collision settings, limits, capacities, warmup,
and destruction. Call `planner.destroy()` after all wrapper requests finish.

For YAM, import your combined two-arm URDF into cuRobo, fit and inspect collision
spheres, and use measured base transforms and the actual gripper/camera geometry.
No guessed YAM model or calibration is bundled. Map observations explicitly:

```python
from astrobo.yam import YamJointLayout

layout = YamJointLayout(left=left_model_joint_names, right=right_model_joint_names)
measured_joints = layout.measured_joints(observation.proprioception["joint_pos"])
```

Adapt the observation access to your installed Inspect Robots version. The helper
accepts its 14-value left-then-right packed vector and excludes the two normalized
gripper apertures. Use a model with twelve active arm joints and conservatively
covered locked grippers; active gripper models need a calibrated mapping of their own.

For Inspect Robots, attach `preview.motion_report()` to the next observation's
`extra["motion_report"]` and expose the wrapper methods through your policy's tool
dispatcher. Label this as planning feedback, not a report of physical execution.
Its existing `pre_check` hook only approves/rejects a proposed chunk: it cannot
substitute the path returned by `plan_motion`. Execution integration is separate.

Before any downstream execution, revalidate the measured start and scene, preserve
the returned joint ordering and `dt`, and retain the embodiment's enforcement guards.
Do not send the agent's original straight-line motion after validating a different path.
IK failure means no solution was found, not proven unreachability. Planning success
is conditional on the configured scene and collision model, not a grasp-success claim.

Replace the whole scene through `robot.update_world(scene)` within preallocated
capacities. This serializes solves and scene changes; do not mutate the underlying
planner concurrently. The scene revision is informational, not an execution token.

Run tests from the astrobo repository:

```bash
python -m pytest tests -q
```

## Compatibility

The inspected cuRobo API is NVlabs/curobo commit
`5e4f23d63d9ec58aca4ce8277a75d20f7b02cc7d`. GPU compatibility has not yet
been validated. Install CUDA dependencies using that checkout’s installation guide.
The lightweight CI job checks packaging and dependency-free contracts; the fuller
CPU motion suite requires torch and this cuRobo API.
