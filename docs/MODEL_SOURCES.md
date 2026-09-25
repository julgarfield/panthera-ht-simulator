# Official model provenance and import audit

Inspected before implementation on 2026-09-23. The downloaded files under
`assets/panthera/` are unchanged upstream files, verified against Git blob IDs
and recorded with SHA-256 hashes in `assets/panthera/provenance.json`.

## Which Panthera?

The [linked product page](https://hightorquerobotics.com/Panthera-HT_Hub/index_en.html)
names **Panthera-HT**, six axes, 4.35 kg, and an advertised 860 mm reach. It links to
[HighTorque's Panthera-HT_Main](https://github.com/HighTorque-Robotics/Panthera-HT_Main).
That repository identifies the standard HT SDK. The product page was fetched
directly and its HTML snapshot is in `research/official_page.html`.

The official descriptions were compared, rather than selecting any six-axis arm:

| Official repository | Inspected revision | Finding |
|---|---|---|
| [Panthera-HT_SDK](https://github.com/HighTorque-Robotics/Panthera-HT_SDK/tree/4a267c148cffa56b369222110bcd0b925809e365) | `4a267c148cffa56b369222110bcd0b925809e365` | Standard follower URDF: six revolute joints and `tool_link`. This revision has no separately articulated fingers or camera housing. |
| [Panthera_digital_twin](https://github.com/HighTorque-Robotics/Panthera_digital_twin/tree/86904070bececb728013a38658ef7bd674cbc6a3) | `86904070bececb728013a38658ef7bd674cbc6a3` | Standard follower has identical arm joint origins, axes and limits. Its web viewer connects to a real SDK; not adopted as the physics runtime. Also contains other legacy arm descriptions, deliberately not used. |
| [Panthera-HT_RoboTwin](https://github.com/HighTorque-Robotics/Panthera-HT_RoboTwin/tree/02559b217308956f0ad58d94e01a2862293b12ff) | `02559b217308956f0ad58d94e01a2862293b12ff` | **Selected:** `RoboTwin/assets/embodiments/panthera-6dof/`. Same standard HT arm chain and TCP, with official camera housing and symmetric prismatic fingers. |
| [Panthera-HT_S_SDK](https://github.com/HighTorque-Robotics/Panthera-HT_S_SDK/tree/df7183734b0ab684479552f33a5ee86fb6844a4e) | `df7183734b0ab684479552f33a5ee86fb6844a4e` | Different, shorter arm. Its joint3 origin is −0.15613 m rather than −0.26 m; joint4 X offset 0.13813 m rather than 0.23 m. It ends in `tcp_link` through `tcp_joint` at 0.12233 m, unlike HT's `tool_link` at 0.165 m. **Not selected.** |

The RoboTwin assets are the official simulation configuration corresponding to
the standard HT kinematic chain. They are not claimed to describe every hardware
revision. No dimensions were stretched to match the advertised reach: the URDF
is the source of geometry and kinematics. The website's reach measurement
convention is not defined in the robot files.

## Exact files used

From the pinned RoboTwin directory:

- `panthera_6dof.urdf`: link/joint origins, axes, visual and collision geometry,
  masses, inertia tensors, limits, camera attachment, fingers and tool frame.
- `meshes/base_link.STL`, `link1.STL` through `link6.STL`, `camera_link.STL`,
  `gripper_L_link.STL`, `gripper_R_link.STL`: original full-resolution meshes.
- `panthera_6dof.srdf`: collision exclusions and semantic reference.
- `curobo.yml`: additional self-collision exclusions used by upstream RoboTwin.
- `config.yml`: source reference for home, gripper mapping and frame conventions.
- `collision_panthera.yml`: bundled provenance/reference only; collision spheres
  are not substituted for the robot's mesh collision geometry.
- Root `LICENSE`: MIT, copyright 2026 HighTorque Robotics.

RoboTwin includes changes relative to the SDK snapshot: `link6` mass is 0.30 kg
versus 0.32 kg; the camera and two fingers have their own mass/inertia; meshes
for link2, link3 and link6 differ. The simulation uses the complete RoboTwin set
consistently, without mixing masses or meshes from the SDK. The arm joint frames,
axes, limits and `tool_joint` agree. The digital-twin snapshot lacks a top-level
license in its tree; no software or assets are copied from it into the runtime.

## Source joint definitions

Joint origins below are in their parent link frames, at zero joint angle. All
six arm joints have zero origin RPY in the selected URDF.

| Joint | Origin XYZ (m) | Axis | Limits (rad) | Effort (N·m) | Velocity (rad/s) |
|---|---|---|---|---|---|
| joint1 | 0, 0, 0.0584 | 0, 0, 1 | −2.4 … 2.4 | 21 | 4.2 |
| joint2 | 0.018199, 0, 0.053 | 0, 1, 0 | 0 … 3.2 | 36 | 5 |
| joint3 | −0.26, 0, 0 | 0, −1, 0 | 0 … 4 | 36 | 5 |
| joint4 | 0.23, 0, 0.06 | 0, −1, 0 | −1.6 … 1.6 | 21 | 4.2 |
| joint5 | 0.07, 0, 0.036319 | 0, 0, −1 | −1.7 … 1.7 | 10 | 3.7 |
| joint6 | 0.02345, 0, −0.039 | 1, 0, 0 | −2.5 … 2.5 | 10 | 3.7 |

`tool_link` is fixed 0.165 m along link6's +X, with unchanged orientation.
The gripper joint origins are at `[0.16555, 0, 0]` in link6. Both slide along +Y:
left 0…0.04 m, right −0.04…0 m. The right mimics the left with multiplier −1.
Their source effort is 100 N and velocity 0.1 m/s. Simulated servos are coupled
using that mimic relation and commanded symmetrically.

## Import and numerical handling

1. Original files stay unchanged in `assets/panthera/`.
2. Binary STLs are converted losslessly to OBJ under `build/meshes/`. MuJoCo 3.3.7
   rejects an STL above 200,000 faces; the official link2 has 321,250. Every source
   triangle is retained, with float32 vertex coordinates round-tripped exactly.
   There is no decimation, rescaling, redesign or replacement arm geometry.
3. A generated URDF resolves mesh paths. `fusestatic=false` retains every body
   frame; visual meshes are retained; automatic inertia balancing is disabled.
4. MuJoCo's URDF compiler produces MJCF. Because its XML exporter prints limited
   decimal precision, original origins, axes, limits, full inertia tensors,
   masses and mesh origin transforms are reapplied at full precision before
   compiling the final scene. NumPy's double-precision eigensolver expresses the
   original inertia tensors as principal moments and orientation without
   MuJoCo's lower-precision fast tensor diagonalization.
5. Environment, cameras, mimic constraint, servos and official collision
   exclusions are added to the generated model. Duplicated collision meshes are
   hidden visually but remain active in physics.

MuJoCo uses convex hulls for mesh contacts. Thus visual geometry is exact while
contacts cannot reproduce all concavities in the source STL. This is the engine's
collision model, not a replacement visual robot. No artificial object attachment
is used for grasping. The included contact test lifts the cube through finger
friction and physical contact.

## Explicit simulation assumptions

These are in `config.yaml`, not represented as measured hardware properties:

- Table geometry, object sizes/masses/friction, base mounting pose and lighting.
- Home/work pose `[0, 1.9, 1.7, -1.35, 0, 0]` radians chosen above the table.
  Upstream `config.yml` specifies six zeros as the SDK reset pose. H uses the
  configured work pose; it is not hardware homing or encoder calibration.
- Servo gains, damping, surface friction and teleoperation speeds. URDF effort
  limits cap total actuator effort, including gravity/bias compensation.
- Table camera pose and intrinsics. Wrist housing attachment comes from the URDF;
  the optical lens offset, orientation and field of view are configurable
  assumptions because no calibrated optical frame/intrinsics accompany it.

Kinematics tests independently traverse the official URDF and compare every
body frame at random valid configurations. They also check the limits, masses,
inertia tensors, file checksums and exact STL triangle conversion.
