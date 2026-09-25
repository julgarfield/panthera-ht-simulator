# Dataset schema 1.0

Each press of SPACE starts/stops one independent episode. Episode numbers are
allocated by creating a new directory; existing data is never overwritten.

```text
datasets/episode_0001/
  metadata.json
  episode.h5
```

During recording the file is `episode.partial.h5`; only a successful close
renames it. A crash, writer error or aborted session is not labelled complete.
Disk errors propagate to the application. A bounded writer queue applies
backpressure instead of dropping RGB frames. Completed episodes are accepted by
the replay validator; partial ones require explicit inspection/recovery.

## Time and command alignment

Default physics runs at 240 Hz, controls at 120 Hz, both RGB cameras at 30 Hz.
Rates use integer divisors, so every camera sample corresponds to exactly one
control boundary (every fourth control tick). The two cameras render the same
frozen physics state; no physics step occurs between captures.

- `/timestamps`: episode-relative simulation seconds, starting at 0.
- `/simulation_timestamps`: monotonic simulation time within the scene run.
- `/frame_index`: zero-based image-pair index.
- `/control_index`: index into the full 120 Hz control stream.
- `/wall_elapsed`: monotonic wall time at frame submission, relative to recorder
  creation. Useful for measuring performance; not a sensor exposure timestamp.

At time **t**, state and cameras observe the current state. The action on that
row begins at **t** and applies over **[t, t + 1/control_hz)**. The next three
control actions are in `/control`, not folded into the camera-rate action row.
For learning, use these controls or explicitly define your own action aggregation
over the camera interval. Never assume one 30 Hz action was held for four ticks.

Pausing stops simulation, camera sampling and logging. No repeated observations
are inserted. Reset ends the active episode and begins a new scene clock.
Recording toggles take effect at a camera boundary. Under insufficient hardware
throughput, simulation time slows without skipping timesteps or frames. Check
wall performance in the UI/report rather than inferring it from simulation FPS.

## HDF5 datasets

Let N be the number of camera pairs, M the number of control samples, and B the
number of free objects. Floats are float64 unless stated otherwise.

| Dataset | Shape | Meaning |
|---|---|---|
| `/observations/joint_positions` | N × 6 | Measured arm radians, joint1…joint6 |
| `/observations/joint_velocities` | N × 6 | Measured rad/s |
| `/observations/joint_torques` | N × 6 | Actuator generalized effort evaluated at the observed state; N·m |
| `/observations/end_effector_pose` | N × 7 | tool_link world position XYZ, quaternion **wxyz** |
| `/observations/gripper_positions` | N × 2 | Left/right slider positions, metres |
| `/observations/gripper_velocities` | N × 2 | Left/right slider velocities, m/s |
| `/observations/object_poses` | N × B × 7 | World XYZ + wxyz, object order in metadata |
| `/observations/qpos`, `/observations/qvel` | N × nq, N × nv | Full MuJoCo generalized state for scene replay |
| `/observations/images/table_camera` | N × H × W × 3 | uint8 RGB, top row first |
| `/observations/images/wrist_camera` | N × H × W × 3 | uint8 RGB, top row first |
| `/observations/camera_extrinsics/{table_camera,wrist_camera}` | N × 4 × 4 | World-from-camera transform, OpenGL optical axes |
| `/actions/mode` | N | uint8: joint=0, Cartesian=1, home=2 |
| `/actions/joint_velocity_command` | N × 6 | Requested keyboard/policy rad/s |
| `/actions/cartesian_twist_command` | N × 6 | Requested world vx,vy,vz,wx,wy,wz in m/s and rad/s |
| `/actions/gripper_velocity_command` | N | Requested left-finger m/s, positive=open |
| `/actions/joint_position_target` | N × 8 | Applied bounded position setpoint: six arm joints, left/right fingers |
| `/control/*` | M × … | Same state and action fields at 120 Hz, plus timestamps and index; no image arrays |

Raw user commands and resulting bounded position targets are both retained.
In Cartesian mode the target captures the IK result. In home mode the zero raw
twist/velocity fields are disambiguated by the mode and the resulting targets.
Gripper slider position is not the calibrated jaw aperture: finger mesh geometry
also contributes to the actual gap. No real-robot gripper calibration is inferred.

MuJoCo's servo damping is integrated implicitly. At every physics substep the
position target receives `qfrc_bias/kp` compensation; total servo effort is capped
at the source URDF limit. This uses the configuration stored with the episode.
The observed torque belongs to the current observed state and previous motor
evaluation, not the future interval's torque.

## Cameras and frames

World coordinates use metres with +Z up. `tool_link` and robot link frames are
the official URDF frames. All recorded quaternions use **w,x,y,z** ordering.

`world_from_camera` uses local +X right, +Y up, −Z forward (OpenGL).
To obtain world-from-OpenCV-camera (+X right, +Y down, +Z forward), postmultiply
by `diag(1,-1,-1,1)`. Pixels are RGB, not BGR. Images have no simulated lens
distortion. Metadata stores each intrinsic matrix K and camera configuration.
The vertical FOV defines `fx=fy=0.5*height/tan(fovy/2)` and the pixel-center
principal point is `((width-1)/2, (height-1)/2)`.

## Export and replay

```powershell
.\.venv\Scripts\python.exe main.py --validate datasets/episode_0001
.\.venv\Scripts\python.exe main.py --export datasets/episode_0001
.\.venv\Scripts\python.exe main.py --replay datasets/episode_0001
```

Export creates an `export/` directory with `states.csv`, `actions.csv`,
`control.csv`, `camera_poses.csv`, `metadata.json`, and both PNG image directories.
CSV field names use HDF5 paths and zero-based flattened component indices.
Object order and pose component order are described above. PNG filenames match
`frame_index`. Export refuses a nonempty destination.

Replay restores the measured 120 Hz arm, gripper and object trajectory, with
recorded 30 Hz RGB shown alongside it by default. It is kinematic trajectory
playback, not a claim that contacts can be reproduced by open-loop action
re-execution. `--live-replay-cameras` renders the restored scene instead;
`--speed 0.5` slows playback; `--loop` repeats it. No interpolation of free-joint
quaternions is needed because stored control states are used directly.

## Policy interface

```python
from panthera.config import load_config
from panthera.environment import SimulatedPanthera
from panthera.interfaces import Action

env = SimulatedPanthera(load_config())
observation = env.get_observation()
action = Action(joint_velocity=[0, 0, 0, 0, 0, 0])
observation = env.step(action)  # advances one 120 Hz control interval
```

`Renderer.capture()` updates `table_camera`, `wrist_camera` and
`image_timestamp` in the environment's observation cache. At non-camera control
ticks these are the latest images, with their explicit older timestamp. Headless
physics-only use need not create a renderer. Recording receives copies at the
capture boundary, so it never mistakes cached images for new ones.

Keyboard input, execution, rendering and recording have separate modules. A
future `RealPanthera` can implement `RobotInterface` and supply matching state
and image fields using the official SDK. This project contains no hardware
connection code. The schema is general-purpose; it is not automatically a LeRobot
or any specific VLA training package's dataset without an appropriate adapter.
