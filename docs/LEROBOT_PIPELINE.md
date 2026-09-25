# Panthera pick/place dataset contract

API inspection and native Windows validation: 2026-09-25. Export dependency:
**LeRobot 0.6.1**, pinned in `requirements-export.txt`. No training, tokenizer,
model-weight download, or data upload is part of these commands.

Official references inspected:

- [LeRobot dataset API](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/lerobot_dataset.py):
  `LeRobotDataset.create`, `add_frame`, `save_episode`, `finalize`, and local loading.
- [Dataset v3 documentation](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3).
- [π₀.₅ configuration](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/pi05/configuration_pi05.py):
  feature dictionaries, default 32 state/action dimensions, absolute actions,
  50-frame action horizon and quantile normalization.
- [π₀.₅ policy documentation](https://huggingface.co/docs/lerobot/en/pi05):
  two-camera inputs and the need for q01/q99 statistics or explicit mean/std normalization.

## Task and labels

The instruction is `Pick up the red cube and place it in the green target area.`
The target is a 130 mm square visual site on the table, with no collision body.
The six-axis official robot assets, limits and frames are unchanged.

`--seed N` chooses deterministic cube/target XY positions. Reset increments N.
Ranges and criteria live under `task` in `config.yaml`. Sampling rejects layouts
that overlap the target/cube or the fixed distractors. The cube starts upright;
lighting, cameras, object shape and material are not randomized. Each episode
records its seed, actual initial cube/target positions and criterion values.

Success is evaluated every 120 Hz step, using simulation geometry and contacts:

1. During this recording the cube's lowest corner must rise at least 40 mm above
   the table while touching a gripper finger.
2. All eight cube corners must be inside the target's XY bounds; the bottom must
   be within 6 mm of the table and have table contact, with no robot contact.
3. Linear speed must stay below 25 mm/s and angular speed below 0.25 rad/s for
   0.5 continuous seconds.

Success reflects the **final current placement**, not a sticky award: moving the
cube out again revokes it. Starting a new recording clears prior lift/dwell
history. Simply sliding the cube into the target does not qualify. `status`
describes file completion separately from `task.success`/`task.outcome`.
Failures and partial files are retained; neither is silently promoted to success.

## State, action and exact timing

Contract identifier: `panthera.absolute_targets_4x7.v1`.

| Feature | Shape | Meaning |
|---|---|---|
| `observation.state` | 7 | Six measured joint radians, then left slider metres |
| `observation.images.table_camera` | H × W × 3 | Synchronized uint8 RGB in storage |
| `observation.images.wrist_camera` | H × W × 3 | Synchronized uint8 RGB in storage |
| `action` | 28 | Four chronological seven-value absolute target sets |
| `source.control_index` | 4 | Exact original episode-relative 120 Hz control indices |
| `source.control_timestamps` | 4 | Corresponding original simulation seconds |
| `source.seed` | 1 | Layout seed |
| `next.done` | 1 | True only on the episode's final action block |
| `next.success` | 1 | True on that terminal block only if the trial succeeded |
| `task` (loader) / `task_index` (storage) | text / integer | LeRobot task text mapping |

LeRobot additionally creates `timestamp`, `frame_index`, `episode_index`, and
global `index`. Its loader returns RGB as CHW float32 in [0,1]. State/actions
are float32; original HDF5 targets remain float64. Floating features, including
source timestamps, are loaded as float32 by LeRobot, so exact integer control
indices are also stored and checked. Both cameras are rendered from the same
frozen state before any physics advancement.

For frame i at t=i/30, action values in row-major order are:

```text
[q1..q6, g] at t            (control 4i)
[q1..q6, g] at t + 1/120    (control 4i+1)
[q1..q6, g] at t + 2/120    (control 4i+2)
[q1..q6, g] at t + 3/120    (control 4i+3)
```

Each is the bounded **resolved position target**, after keyboard velocity
integration, release behavior, IK, target-lead and joint-limit guards. It is not
a measured future position. g is the left slider's metre position target, with
right target = -g. It is not a calibrated physical jaw width. No averaging,
decimation, interpolation, one-tick shift, terminal padding or dropped tail is
used. A source with M != 4N is rejected if selected for export.

The float32 conversion is the only intentional numeric precision reduction;
lossless image features preserve every RGB pixel. The exporter retains source
hashes, all task metadata and reasons for exclusions in `panthera_manifest.json`.
It writes to a new `.partial` output first, finalizes and validates before rename,
then verifies loading after relocation. A failed export keeps its partial output
for diagnosis; choose a new destination to retry. HDF5 sources are opened read-only.

## Rollout interface

`PolicyController.observe(renderer)` captures fresh synchronized HWC uint8 images
and a seven-value state with task text. Apply the eventual LeRobot processor's
batching, CHW conversion, image resizing and normalization consistently with
training. Feed **denormalized, unpadded** 28-value outputs to the simulator:

```python
from panthera.policy import PolicyController

controller = PolicyController(env)
observation = controller.observe(renderer)  # before inference
# action = your policy + its trained pre/postprocessors: shape (28,)
next_observation, executed_targets = controller.step(action)
```

`step` advances four controls, with two physics steps per control in the default
configuration. It uses the existing PD/bias servo and official position/effort
limits plus the same target-lead guard. `executed_targets` exposes any clipping.
Inference itself does not advance simulation time. This is a simulation adapter,
not a real-time scheduler or hardware safety controller. Observe again before
the next inference; render is not implied by `step`. Do not feed stale cached
camera images from arbitrary 120 Hz ticks.

For LeRobot π₀.₅, resolve the two camera input features and `(7,)` state from the
dataset, with an `(28,)` action output. The current default padded dimensions
are both 32, so this fits without changing the backbone dimension. Leave
`use_relative_actions=False`: the stock relative-state subtraction is not a
mapping from seven state values to these four target sets. A 50-frame predicted
chunk represents 50 camera intervals (200 control ticks, 1.667 s), not 50 low-level
commands. Each output row still goes through `PolicyController.step`. The
integration test constructs `PI05Config` with these features and checks boundary
padding without instantiating a model. The native writer supplies the q01/q99
statistics, and validation checks they exist and are finite.

Shape/API compatibility is not proof of training quality. This 28D action layout
differs from common seven-value pretraining actions; successful adaptation needs
real fine-tuning and unseen-seed evaluation. No inference or learning quality is
claimed by these tests.

## Reproducible checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv-export\Scripts\python.exe -m pip install pytest==8.4.2
.\.venv-export\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe tools/record_smoke_trial.py --output validation_runs/task_smoke --seed 0
.\.venv-export\Scripts\python.exe -m panthera.lerobot_export --source validation_runs/task_smoke --output validation_runs/lerobot_smoke
```

The scripted 20-second seed-0 trial is a regression fixture that uses actual
gripper contact. It does not attach/teleport the cube or override its success
label, and is not a general demonstration generator. Human trials should include
varied approaches and recovery choices rather than copies of this trajectory.
Normal keyboard collection does not import this script.

## Physical-arm transfer still required

- An official-SDK backend must map six radians and the left-slider metre target
  to calibrated hardware encoders/gripper commands, preserving joint order,
  signs, coupling and the four-point 120 Hz schedule. No hardware backend exists.
- Real inference/transport latency, clock synchronization, missed-command
  watchdogs, conservative velocity/acceleration limits and emergency stops need
  hardware-specific implementation. Simulator target-lead clipping is not a
  substitute for those safeguards; it does not establish a hardware slew limit.
- Calibrate intrinsics, wrist optical pose, table/world alignment, actual jaw
  aperture and contact/friction/gains. MuJoCo convex mesh collisions and tuned
  PD/bias compensation are not measured real-arm dynamics.
- The physical target and cube must be observed/calibrated for real success
  evaluation. Simulation-only ground-truth geometry and contact flags are not
  policy inputs and cannot serve as a real-world success detector.
- Collect real-arm demonstrations and evaluate held-out scene configurations.
  Seeds alone do not bridge differences in lighting, appearance, latency and
  mechanics. Keep training/evaluation seed sets and recording folders separate.
