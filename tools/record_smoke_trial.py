"""Record a deterministic contact-based integration trial, not a trained policy.

Seed 0 is the regression fixture. Other layouts are not guaranteed to succeed.
No attachment constraints, object teleporting, or manual success labels are used.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from panthera.config import load_config
from panthera.environment import SimulatedPanthera
from panthera.interfaces import Action
from panthera.recorder import EpisodeRecorder
from panthera.rendering import Renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    env = SimulatedPanthera(load_config())
    env.reset(seed=args.seed)
    renderer = Renderer(env, visible=False)
    recorder = EpisodeRecorder(env.cfg, env, renderer.gpu, args.output)
    x, y = env.task.cube_start[:2]
    a, b = env.task.target[:2]
    aborted = True
    try:
        for stage, goal, grip, seconds in [
            ('approach', [x,y,.764], 0, 4), ('close', [x,y,.764], -.025, 2),
            ('lift', [x,y,.94], 0, 4), ('transfer', [a,b,.94], 0, 4),
            ('lower', [a,b,.78], 0, 4), ('release', [a,b,.78], .025, 2),
        ]:
            for _ in range(int(seconds/env.dt)):
                observation = env.get_observation()
                error = np.asarray(goal)-observation['end_effector_pose'][:3]
                action = Action(mode='cartesian', cartesian_twist=np.r_[np.clip(error*2,-.06,.06),0,0,0], gripper_velocity=grip)
                targets = env.apply_action(action)
                recorder.record_control(observation, action, targets)
                if env.tick % 4 == 0:
                    images, poses = renderer.capture()
                    recorder.record_frame(observation, action, targets, images, poses)
                env.advance()
            print(f'{stage}: {env.task.metadata()["outcome"]}, {env.task.metrics}', flush=True)
        aborted = False
    finally:
        try:
            print(f'Saved {recorder.close(aborted=aborted)}', flush=True)
        finally:
            renderer.close()
    if not env.task.success:
        raise SystemExit('Smoke trial failed the objective success check; HDF5 retained.')


if __name__ == '__main__':
    main()
