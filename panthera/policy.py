"""28D absolute target blocks: four 7D controls per 30 Hz policy observation."""
import numpy as np

ACTION_CONTRACT = 'panthera.absolute_targets_4x7.v1'
STATE_NAMES = [f'joint{i}' for i in range(1, 7)] + ['gripper']
ACTION_NAMES = [f'tick{k}.{name}' for k in range(4) for name in STATE_NAMES]


def policy_state(observation):
    """Six measured radians + left slider metres; not a calibrated jaw aperture."""
    return np.r_[observation['joint_positions'], observation['gripper_positions'][0]].astype(np.float32)


def pack_targets(targets):
    targets = np.asarray(targets)
    if targets.shape != (4, 8) or not np.isfinite(targets).all():
        raise ValueError('Expected four finite eight-actuator target rows')
    if not np.allclose(targets[:, 7], -targets[:, 6], atol=1e-9, rtol=0):
        raise ValueError('Gripper targets are not coupled')
    return targets[:, :7].astype(np.float32).reshape(28)


class PolicyController:
    """Simulation-time adapter; accepts denormalized actions, never model weights.

    Each step executes four controls then returns state at the NEXT camera time.
    Call observe(renderer) before inference to get fresh synchronized RGB/state.
    """
    def __init__(self, env):
        if env.cfg['simulation']['control_hz'] != 120 or any(env.cfg[n]['fps'] != 30 for n in ('table_camera', 'wrist_camera')):
            raise ValueError('This action contract requires 120 Hz control and 30 Hz cameras')
        self.env = env

    def observe(self, renderer):
        images, _ = renderer.capture()
        obs = self.env.get_observation()
        return {'observation.state': policy_state(obs),
                **{f'observation.images.{n}': images[n] for n in ('table_camera', 'wrist_camera')},
                'task': self.env.task.metadata()['instruction']}

    def step(self, action):
        block = np.asarray(action, dtype=float)
        if block.shape != (28,) or not np.isfinite(block).all():
            raise ValueError('Policy action must be a finite, denormalized (28,) vector')
        executed = []
        for target in block.reshape(4, 7):
            executed.append(self.env.apply_position_target(target))
            self.env.advance()
        return self.env.get_observation(), np.asarray(executed)
