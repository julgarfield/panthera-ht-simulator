"""Simulation-only, seed-reproducible red-cube pick/place task."""
from itertools import product
import numpy as np

INSTRUCTION = 'Pick up the red cube and place it in the green target area.'
DEFAULT_TASK = {
    'seed': 0, 'cube_x': [.30, .36], 'cube_y': [-.055, .04],
    'target_x': [.40, .47], 'target_y': [-.10, .02],
    'target_half_size': .065, 'min_separation': .135,
    'lift_height': .04, 'settle_seconds': .5,
    'linear_speed': .025, 'angular_speed': .25, 'surface_tolerance': .006,
}


def validate_task_config(cfg):
    c = {**DEFAULT_TASK, **cfg.get('task', {})}
    if type(c['seed']) is not int or not 0 <= c['seed'] < np.iinfo(np.int64).max:
        raise ValueError('task.seed must be a non-negative signed 64-bit integer')
    for key in ('target_half_size', 'min_separation', 'lift_height', 'settle_seconds',
                'linear_speed', 'angular_speed', 'surface_tolerance'):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'task.{key} must be finite and positive')
    cube = next((o for o in cfg['environment']['objects'] if o['name'] == 'cube'), None)
    if cube is None or cube['type'] != 'box':
        raise ValueError('The pick/place task requires the box object named cube')
    for obj in ('cube', 'target'):
        for i, axis in enumerate(('x', 'y')):
            bounds = np.asarray(c[f'{obj}_{axis}'], dtype=float)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
                raise ValueError(f'Invalid task.{obj}_{axis} bounds')
            center = cfg['environment']['table_position'][i]
            half = cfg['environment']['table_size'][i]/2
            margin = cube['size'][i] if obj == 'cube' else c['target_half_size']
            if bounds[0]-margin < center-half or bounds[1]+margin > center+half:
                raise ValueError(f'task.{obj}_{axis} extends outside the table')


class PickPlaceTask:
    def __init__(self, env):
        self.env = env
        self.cfg = {**DEFAULT_TASK, **env.cfg.get('task', {})}
        self.cube = env.model.body('cube').id
        self.cube_geom = env.model.geom('cube_geom').id
        self.table_geom = env.model.geom('table').id
        self.cube_joint = env.model.joint('cube_free').id
        self.half_size = env.model.geom_size[self.cube_geom].copy()
        self.table_z = float(env.model.geom_pos[self.table_geom, 2] + env.model.geom_size[self.table_geom, 2])
        self.robot_bodies = set()
        for bid in range(1, env.model.nbody):
            parent = bid
            while parent:
                if parent == env.model.body('base_link').id:
                    self.robot_bodies.add(bid)
                    break
                parent = int(env.model.body_parentid[parent])
        self.finger_bodies = {int(env.model.jnt_bodyid[j]) for j in env.jids[6:]}
        self.seed = None

    def reset(self, seed=None):
        if seed is None:
            seed = self.cfg['seed'] if self.seed is None else self.seed + 1
        if not isinstance(seed, (int, np.integer)) or not 0 <= seed < np.iinfo(np.int64).max:
            raise ValueError('Task seed must be a non-negative integer')
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        c = self.cfg
        for _ in range(10000):
            cube_xy = rng.uniform([c['cube_x'][0], c['cube_y'][0]], [c['cube_x'][1], c['cube_y'][1]])
            target_xy = rng.uniform([c['target_x'][0], c['target_y'][0]], [c['target_x'][1], c['target_y'][1]])
            if np.linalg.norm(cube_xy-target_xy) < c['min_separation']:
                continue
            if np.all(np.abs(cube_xy-target_xy) <= c['target_half_size']+self.half_size[:2]+.005):
                continue
            # Keep the target and cube clear of the two stationary distractors.
            if any(np.linalg.norm(target_xy-np.array(o['position'][:2])) < c['target_half_size']*1.42 + max(o['size'][:2]) + .01
                   or np.linalg.norm(cube_xy-np.array(o['position'][:2])) < np.linalg.norm(self.half_size[:2]) + max(o['size'][:2]) + .01
                   for o in self.env.cfg['environment']['objects'] if o['name'] != 'cube'):
                continue
            break
        else:
            raise ValueError('Task position ranges cannot produce a separated layout')
        self.cube_start = np.r_[cube_xy, self.table_z + self.half_size[2] + .002]
        self.target = np.r_[target_xy, self.table_z + .0005]
        qid = self.env.model.jnt_qposadr[self.cube_joint]
        self.env.data.qpos[qid:qid+7] = np.r_[self.cube_start, 1., 0., 0., 0.]
        self.env.model.site_pos[self.env.model.site('placement_target').id] = self.target
        self.begin_episode()

    def begin_episode(self):
        self.lifted = False
        self.settled_ticks = 0
        self.success = False
        self.metrics = {}

    def update(self):
        env, c = self.env, self.cfg
        # All eight corners, including cube orientation, must fit inside the target.
        corners = np.array(list(product((-1., 1.), repeat=3))) * self.half_size
        corners = corners @ env.data.xmat[self.cube].reshape(3, 3).T + env.data.xpos[self.cube]
        bottom = float(corners[:, 2].min())
        robot_contact = finger_contact = table_contact = False
        for contact in env.data.contact[:env.data.ncon]:
            if self.cube_geom not in (contact.geom1, contact.geom2) or contact.dist > .001:
                continue
            other = contact.geom2 if contact.geom1 == self.cube_geom else contact.geom1
            body = int(env.model.geom_bodyid[other])
            robot_contact |= body in self.robot_bodies
            finger_contact |= body in self.finger_bodies
            table_contact |= other == self.table_geom
        if bottom >= self.table_z+c['lift_height'] and finger_contact:
            self.lifted = True
        inside = bool(np.all(np.abs(corners[:, :2]-self.target[:2]) <= c['target_half_size']))
        vid = env.model.jnt_dofadr[self.cube_joint]
        velocity = env.data.qvel[vid:vid+6]
        stable = bool(np.linalg.norm(velocity[:3]) < c['linear_speed'] and np.linalg.norm(velocity[3:]) < c['angular_speed'])
        placed = self.lifted and inside and abs(bottom-self.table_z) < c['surface_tolerance'] and table_contact and not robot_contact and stable
        self.settled_ticks = self.settled_ticks + 1 if placed else 0
        self.success = self.settled_ticks >= int(np.ceil(c['settle_seconds']/env.dt))
        self.metrics = dict(lifted=self.lifted, inside_target=inside, table_contact=table_contact,
                            robot_contact=robot_contact, stable=stable, bottom_z=bottom,
                            settled_seconds=self.settled_ticks*env.dt)

    def metadata(self):
        return dict(instruction=INSTRUCTION, seed=self.seed, cube_start=self.cube_start.tolist(),
                    target_center=self.target.tolist(), criteria={**self.cfg, 'seed': self.seed},
                    success=bool(self.success), outcome='success' if self.success else 'failure',
                    metrics=self.metrics.copy())
