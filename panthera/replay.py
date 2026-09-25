from pathlib import Path
from copy import deepcopy
import csv
import json
import h5py
import numpy as np
from PIL import Image
from .config import ROOT, validate_config


class EpisodeReplay:
    def __init__(self, path):
        path = Path(path).resolve()
        self.path = path.parent if path.is_file() else path
        self.metadata = json.loads((self.path/'metadata.json').read_text(encoding='utf-8'))
        if self.metadata.get('status') != 'complete':
            raise ValueError('This episode was not completed. Inspect its partial file before attempting recovery.')
        self.file = h5py.File(self.path/'episode.h5', 'r')
        try:
            self.validate()
        except Exception:
            self.file.close()
            raise
        self.control_times = self.file['control/timestamps'][:]
        self.frame_times = self.file['timestamps'][:]
        self.qpos = self.file['control/qpos'][:]
        self.qvel = self.file['control/qvel'][:]

    def validate(self):
        f = self.file
        if self.metadata['schema_version'] not in ('1.0', '1.1'):
            raise ValueError('Unsupported dataset schema')
        count, controls = len(f['timestamps']), len(f['control/timestamps'])
        if not count or not controls:
            raise ValueError('Episode contains no samples')
        errors = []
        def check(name, obj):
            if isinstance(obj, h5py.Dataset):
                expected = controls if name.startswith('control/') else count
                if len(obj) != expected:
                    errors.append(f'{name}: {len(obj)} != {expected}')
        f.visititems(check)
        ct = f['control/timestamps'][:]
        ft = f['timestamps'][:]
        ids = f['control_index'][:]
        if not np.allclose(np.diff(ct), 1/self.metadata['control_hz'], atol=1e-9, rtol=0):
            errors.append('Control timestamps are not uniformly spaced')
        if not np.allclose(np.diff(ft), 1/self.metadata['camera_fps'], atol=1e-9, rtol=0):
            errors.append('Camera timestamps are not uniformly spaced')
        if np.any(ids < 0) or np.any(ids >= controls):
            errors.append('Invalid frame to control indices')
        elif not np.allclose(ft, ct[ids], atol=1e-9, rtol=0):
            errors.append('Camera/control timestamps do not align')
        else:
            for key in ('joint_positions', 'joint_velocities', 'qpos', 'qvel', 'end_effector_pose', 'gripper_positions'):
                if not np.array_equal(f[f'observations/{key}'][:], f[f'control/{key}'][:][ids]):
                    errors.append(f'Frame/control state mismatch: {key}')
            for key in ('mode', 'joint_position_target', 'joint_velocity_command', 'cartesian_twist_command', 'gripper_velocity_command'):
                if not np.array_equal(f[f'actions/{key}'][:], f[f'control/{key}'][:][ids]):
                    errors.append(f'Frame/control action mismatch: {key}')
        for camera in ('table_camera', 'wrist_camera'):
            cam = self.metadata['configuration'][camera]
            ds = f[f'observations/images/{camera}']
            if ds.shape != (count, cam['height'], cam['width'], 3) or ds.dtype != np.uint8:
                errors.append(f'Invalid {camera} RGB dimensions')
        if count != self.metadata['number_of_samples'] or controls != self.metadata['number_of_control_samples']:
            errors.append('Metadata sample counts disagree')
        if errors:
            raise ValueError('; '.join(errors))

    def config(self):
        cfg = deepcopy(self.metadata['configuration'])
        if 'task' in self.metadata:
            cfg['task'] = deepcopy(self.metadata['task']['criteria'])
            cfg['task']['seed'] = self.metadata['task']['seed']
        cfg['_root'] = str(ROOT)
        validate_config(cfg)
        return cfg

    def images(self, time):
        i = int(np.clip(np.searchsorted(self.frame_times, time, side='right')-1, 0, len(self.frame_times)-1))
        return {name: self.file[f'observations/images/{name}'][i] for name in ('table_camera', 'wrist_camera')}

    def close(self):
        self.file.close()


def export_episode(path, destination=None):
    episode = EpisodeReplay(path)
    try:
        out = Path(destination) if destination else episode.path/'export'
        if out.exists() and any(out.iterdir()):
            raise FileExistsError(f'Export folder is not empty: {out}')
        out.mkdir(parents=True, exist_ok=True)
        (out/'metadata.json').write_text(json.dumps(episode.metadata, indent=2), encoding='utf-8')
        f = episode.file
        for name in ('table_camera', 'wrist_camera'):
            folder = out/name
            folder.mkdir(exist_ok=True)
            for i, frame in enumerate(f[f'observations/images/{name}']):
                Image.fromarray(frame).save(folder/f'{i:06d}.png')
        tables = {
            'states.csv': ['timestamps', 'simulation_timestamps', 'frame_index', 'control_index'] +
                          [f'observations/{key}' for key in ('joint_positions', 'joint_velocities', 'joint_torques', 'end_effector_pose', 'gripper_positions', 'object_poses')],
            'actions.csv': ['timestamps', 'frame_index', 'control_index'] + [f'actions/{key}' for key in ('mode', 'joint_velocity_command', 'cartesian_twist_command', 'gripper_velocity_command', 'joint_position_target')],
            'control.csv': [f'control/{key}' for key in ('timestamps', 'simulation_timestamps', 'index', 'joint_positions', 'joint_velocities', 'end_effector_pose', 'gripper_positions', 'mode', 'joint_velocity_command', 'cartesian_twist_command', 'gripper_velocity_command', 'joint_position_target')],
            'camera_poses.csv': ['timestamps', 'frame_index', 'observations/camera_extrinsics/table_camera', 'observations/camera_extrinsics/wrist_camera'],
        }
        for name, paths in tables.items():
            with (out/name).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                headers = []
                for p in paths:
                    shape = f[p].shape[1:]
                    headers.extend([p] if not shape else [f'{p}[{i}]' for i in range(int(np.prod(shape)))])
                writer.writerow(headers)
                for row in range(len(f[paths[0]])):
                    writer.writerow(np.concatenate([np.asarray(f[p][row]).reshape(-1) for p in paths]))
        return out
    finally:
        episode.close()
