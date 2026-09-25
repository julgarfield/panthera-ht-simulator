import mujoco
import numpy as np
from .model import build_model
from .interfaces import Action


class SimulatedPanthera:
    """Execution backend, independent of keyboards, graphics windows, and storage."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, source_limits = build_model(cfg)
        self.data = mujoco.MjData(self.model)
        self.arm_names = [f'joint{i}' for i in range(1, 7)]
        self.names = self.arm_names + ['gripper_L_joint', 'gripper_R_joint']
        self.jids = np.array([self.model.joint(n).id for n in self.names])
        self.qids = self.model.jnt_qposadr[self.jids]
        self.dids = self.model.jnt_dofadr[self.jids]
        self.aids = np.array([self.model.actuator(f'{n}_servo').id for n in self.names])
        self.lower = self.model.jnt_range[self.jids, 0].copy()
        self.upper = self.model.jnt_range[self.jids, 1].copy()
        self.max_velocity = np.array([float(source_limits[n]['velocity']) for n in self.names])
        self.max_effort = np.array([float(source_limits[n]['effort']) for n in self.names])
        self.kp = np.r_[cfg['robot']['kp'], cfg['robot']['gripper_kp'], cfg['robot']['gripper_kp']]
        self.kd = np.r_[cfg['robot']['kd'], cfg['robot']['gripper_kd'], cfg['robot']['gripper_kd']]
        self.tool_id = self.model.body(cfg['robot']['end_effector_link']).id
        self.object_ids = [self.model.body(o['name']).id for o in cfg['environment']['objects']]
        self.dt = 1/cfg['simulation']['control_hz']
        self.substeps = cfg['simulation']['physics_hz']//cfg['simulation']['control_hz']
        self.tick = 0
        self.targets = np.zeros(8)
        self.previous_velocity = np.zeros(8)
        self.last_action = Action()
        self.last_ik_residual = 0.0
        self.latest_images = {}
        self.latest_image_timestamp = None
        self.limit_corrections = 0
        self.reset()

    @property
    def timestamp(self):
        return self.tick / self.cfg['simulation']['control_hz']

    def reset(self):
        home = np.asarray(self.cfg['robot']['home_configuration'], dtype=float)
        if np.any(home < self.lower[:6]) or np.any(home > self.upper[:6]):
            raise ValueError('Configured home violates official joint limits')
        opening = self.cfg['robot']['gripper_open']
        if not 0 <= opening <= self.upper[6]:
            raise ValueError('Invalid gripper home opening')
        mujoco.mj_resetData(self.model, self.data)
        self.targets[:] = np.r_[home, opening, -opening]
        self.data.qpos[self.qids] = self.targets
        self.data.ctrl[:] = 0
        self.previous_velocity[:] = 0
        self.tick = 0
        self.limit_corrections = 0
        self.latest_images = {}
        self.latest_image_timestamp = None
        mujoco.mj_forward(self.model, self.data)

    def hold(self):
        self.targets[:6] = np.clip(self.data.qpos[self.qids[:6]], self.lower[:6], self.upper[:6])
        self.previous_velocity[:] = 0
        self.last_action = Action()

    def apply_action(self, action):
        action.validate()
        controls = self.cfg['controls']
        q = self.data.qpos[self.qids].copy()
        if action.mode == 'cartesian':
            jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.tool_id)
            jac = np.vstack([jacp[:, self.dids[:6]], jacr[:, self.dids[:6]]])
            twist = action.cartesian_twist.copy()
            twist[:3] = np.clip(twist[:3], -2*controls['cartesian_translation_speed'], 2*controls['cartesian_translation_speed'])
            twist[3:] = np.clip(twist[3:], -2*controls['cartesian_rotation_speed'], 2*controls['cartesian_rotation_speed'])
            # Damped resolved-rate IK has no accumulated unreachable pose target.
            velocity = jac.T @ np.linalg.solve(jac @ jac.T + controls['ik_damping']**2*np.eye(6), twist)
            self.last_ik_residual = float(np.linalg.norm(jac @ velocity - twist))
        elif action.mode == 'home':
            velocity = (np.asarray(self.cfg['robot']['home_configuration'])-q[:6])*2.0
            self.last_ik_residual = 0.0
        else:
            velocity = action.joint_velocity.copy()
            self.last_ik_residual = 0.0
        speed = np.minimum(self.max_velocity[:6], 2*controls['joint_speed'])
        velocity = np.clip(velocity, -speed, speed)
        grip_speed = min(2*controls['gripper_speed'], self.max_velocity[6], self.max_velocity[7])
        grip = float(np.clip(action.gripper_velocity, -grip_speed, grip_speed))
        velocities = np.r_[velocity, grip, -grip]
        released = (np.abs(velocities) < 1e-10) & (np.abs(self.previous_velocity) >= 1e-10)
        # Keep the gripper's closure setpoint on release so contact pressure is
        # maintained. Resetting it to measured aperture would let a grasp slip.
        released[6:] = False
        self.targets[released] = q[released]
        self.targets += velocities*self.dt
        # Prevent a blocked arm from accumulating a large stored position command.
        lead = np.r_[np.full(6, controls['max_target_lead']), .015, .015]
        self.targets = np.clip(self.targets, q-lead, q+lead)
        self.targets = np.clip(self.targets, self.lower, self.upper)
        self.targets[7] = -self.targets[6]
        self.previous_velocity = velocities
        self.last_action = action
        return self.targets.copy()

    def advance(self):
        for _ in range(self.substeps):
            mujoco.mj_forward(self.model, self.data)
            self.data.ctrl[self.aids] = self.targets+self.data.qfrc_bias[self.dids]/self.kp
            mujoco.mj_step(self.model, self.data)
            # MuJoCo limits are compliant. Enforce exact source ranges as a final guard.
            q = self.data.qpos[self.qids]
            clipped = np.clip(q, self.lower, self.upper)
            outside = np.abs(q-clipped) > 1e-12
            if outside.any():
                self.data.qpos[self.qids] = clipped
                self.data.qvel[self.dids[outside]] = 0
                self.limit_corrections += int(outside.sum())
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise RuntimeError('Simulation produced a non-finite state; recording stopped')
        self.tick += 1
        # Position-dependent fields after mj_step can lag qpos by one substep.
        self.data.time = self.timestamp
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        self.apply_action(action)
        self.advance()
        return self.get_observation()

    def get_observation(self):
        q, v = self.data.qpos[self.qids], self.data.qvel[self.dids]
        return {
            'timestamp': self.timestamp,
            'joint_positions': q[:6].copy(), 'joint_velocities': v[:6].copy(),
            'joint_torques': self.data.qfrc_actuator[self.dids[:6]].copy(),
            'end_effector_pose': np.r_[self.data.xpos[self.tool_id], self.data.xquat[self.tool_id]],
            'gripper_positions': q[6:].copy(), 'gripper_velocities': v[6:].copy(),
            'object_poses': np.array([np.r_[self.data.xpos[i], self.data.xquat[i]] for i in self.object_ids]),
            'qpos': self.data.qpos.copy(), 'qvel': self.data.qvel.copy(),
            'image_timestamp': self.latest_image_timestamp, **self.latest_images,
        }

    def set_replay_state(self, qpos, qvel, timestamp):
        if np.shape(qpos) != (self.model.nq,) or np.shape(qvel) != (self.model.nv,):
            raise ValueError('Replay state does not match model dimensions')
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.time = timestamp
        self.tick = int(round(timestamp*self.cfg['simulation']['control_hz']))
        self.data.ctrl[:] = 0
        mujoco.mj_forward(self.model, self.data)
