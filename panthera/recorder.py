"""Bounded background HDF5 writer. No frame dropping or unsynchronized reads."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from queue import Queue, Full, Empty
from threading import Thread
import hashlib
import json
import time
import h5py
import numpy as np
from . import __version__
from .config import portable_config, resolve, ROOT

MODE_IDS = {'joint': 0, 'cartesian': 1, 'home': 2}


def camera_intrinsics(camera):
    f = .5*camera['height']/np.tan(np.deg2rad(camera['fov'])/2)
    return [[f, 0, (camera['width']-1)/2], [0, f, (camera['height']-1)/2], [0, 0, 1]]


class EpisodeRecorder:
    def __init__(self, cfg, env, gpu='unknown', dataset_directory=None):
        self.cfg = cfg
        folder = Path(dataset_directory) if dataset_directory else resolve(cfg, cfg['logging']['dataset_directory'])
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(1, 1_000_000):
            self.path = folder / f'episode_{i:04d}'
            try:
                self.path.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError('No unused episode number')
        self.start_time = env.timestamp
        self.wall_start = time.perf_counter()
        self.controls = self.frames = 0
        self.error = None
        self.closed = False
        self.queue = Queue(maxsize=cfg['logging']['queue_size'])
        self.backpressure_seconds = 0.0
        self.metadata = {
            'schema_version': '1.0', 'simulator': 'MuJoCo', 'simulator_version': version('mujoco'),
            'application_version': __version__, 'robot_model': 'Panthera-HT / official RoboTwin panthera-6dof',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'recording',
            'physics_hz': cfg['simulation']['physics_hz'], 'control_hz': cfg['simulation']['control_hz'],
            'camera_fps': cfg['table_camera']['fps'], 'start_simulation_time': self.start_time,
            'joint_names': env.arm_names, 'gripper_joint_names': env.names[6:],
            'object_names': [o['name'] for o in cfg['environment']['objects']],
            'quaternion_order': 'wxyz', 'length_unit': 'metre', 'angle_unit': 'radian', 'time_unit': 'second',
            'action_alignment': 'State and both RGB images at t; action applies on [t, t + 1/control_hz). Full commands in /control.',
            'camera_extrinsics': 'world_from_camera; OpenGL local +X right, +Y up, -Z forward',
            'intrinsics': {n: camera_intrinsics(cfg[n]) for n in ('table_camera', 'wrist_camera')},
            'mode_ids': MODE_IDS, 'gpu': gpu, 'configuration': portable_config(cfg),
            'urdf_sha256': hashlib.sha256(resolve(cfg, cfg['robot']['urdf_path']).read_bytes()).hexdigest(),
            'model_dimensions': {'nq': env.model.nq, 'nv': env.model.nv},
        }
        manifest = ROOT / 'assets/panthera/provenance.json'
        if manifest.exists():
            self.metadata['asset_provenance'] = json.loads(manifest.read_text(encoding='utf-8'))
        self._write_metadata()
        self.thread = Thread(target=self._worker, name='episode-writer', daemon=True)
        self.thread.start()

    def _write_metadata(self):
        temporary = self.path / 'metadata.tmp'
        temporary.write_text(json.dumps(self.metadata, indent=2), encoding='utf-8')
        temporary.replace(self.path / 'metadata.json')

    def check(self):
        if self.error:
            raise RuntimeError(f'Recording failed: {self.error}') from self.error

    def _put(self, item):
        start = time.perf_counter()
        while True:
            self.check()
            if self.closed:
                raise RuntimeError('Recorder is closed')
            try:
                self.queue.put(item, timeout=.05)
                self.backpressure_seconds += time.perf_counter()-start
                return
            except Full:
                continue

    def _row(self, observation, action, target):
        return {
            'timestamps': np.float64(observation['timestamp']-self.start_time),
            'simulation_timestamps': np.float64(observation['timestamp']),
            'joint_positions': observation['joint_positions'].copy(),
            'joint_velocities': observation['joint_velocities'].copy(),
            'joint_torques': observation['joint_torques'].copy(),
            'end_effector_pose': observation['end_effector_pose'].copy(),
            'gripper_positions': observation['gripper_positions'].copy(),
            'gripper_velocities': observation['gripper_velocities'].copy(),
            'object_poses': observation['object_poses'].copy(),
            'qpos': observation['qpos'].copy(), 'qvel': observation['qvel'].copy(),
            'mode': np.uint8(MODE_IDS[action.mode]),
            'joint_velocity_command': action.joint_velocity.copy(),
            'cartesian_twist_command': action.cartesian_twist.copy(),
            'gripper_velocity_command': np.float64(action.gripper_velocity),
            'joint_position_target': np.asarray(target).copy(),
        }

    def record_control(self, observation, action, target):
        row = self._row(observation, action, target)
        row['index'] = np.int64(self.controls)
        self._put(('control', row))
        self.controls += 1

    def record_frame(self, observation, action, target, images, camera_poses):
        row = self._row(observation, action, target)
        row['control_index'] = np.int64(self.controls-1)
        row['frame_index'] = np.int64(self.frames)
        row['wall_elapsed'] = np.float64(time.perf_counter()-self.wall_start)
        for name in ('table_camera', 'wrist_camera'):
            row[f'images/{name}'] = images[name].copy()
            row[f'camera_extrinsics/{name}'] = camera_poses[name].copy()
        self._put(('frames', row))
        self.frames += 1

    @staticmethod
    def _dataset_path(kind, key):
        if kind == 'control':
            return f'control/{key}'
        if key in ('timestamps', 'simulation_timestamps', 'frame_index', 'control_index', 'wall_elapsed'):
            return key
        if key in ('mode', 'joint_velocity_command', 'cartesian_twist_command', 'gripper_velocity_command', 'joint_position_target'):
            return f'actions/{key}'
        return f'observations/{key}'

    def _worker(self):
        try:
            with h5py.File(self.path/'episode.partial.h5', 'w') as file:
                file.attrs['schema_version'] = '1.0'
                file.attrs['metadata_json'] = json.dumps(self.metadata)
                stopping = False
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    batches = {'control': [], 'frames': []}
                    batches[item[0]].append(item[1])
                    deadline = time.perf_counter()+.10
                    # Amortize HDF5 resize/write overhead, especially the 120 Hz
                    # state stream, without unbounded buffering or frame loss.
                    for _ in range(31):
                        remaining = deadline-time.perf_counter()
                        if remaining <= 0:
                            break
                        try:
                            item = self.queue.get(timeout=remaining)
                        except Empty:
                            break
                        if item is None:
                            stopping = True
                            break
                        batches[item[0]].append(item[1])
                    for kind, rows in batches.items():
                        if not rows:
                            continue
                        for key in rows[0]:
                            path = self._dataset_path(kind, key)
                            array = np.stack([row[key] for row in rows])
                            shape = array.shape[1:]
                            if path not in file:
                                image = key.startswith('images/')
                                kwargs = {'compression': self.cfg['logging']['compression']} if image else {}
                                file.create_dataset(path, shape=(0,)+shape, maxshape=(None,)+shape,
                                                    chunks=(1 if image else 128,)+shape, dtype=array.dtype, **kwargs)
                            dataset = file[path]
                            size = len(dataset)
                            dataset.resize(size+len(rows), axis=0)
                            dataset[size:] = array
                    file.flush()
                    if stopping:
                        break
                file.flush()
        except BaseException as error:
            self.error = error

    def close(self, aborted=False):
        if self.closed:
            self.check()
            return self.path
        if self.error is None:
            try:
                self._put(None)
            except RuntimeError:
                pass
        self.thread.join()
        self.closed = True
        wall_seconds = time.perf_counter()-self.wall_start
        self.metadata.update(status='incomplete' if self.error or aborted else 'complete',
                             number_of_samples=self.frames, number_of_control_samples=self.controls,
                             episode_duration=self.controls/self.cfg['simulation']['control_hz'],
                             wall_duration=wall_seconds, measured_capture_fps=self.frames/max(wall_seconds, 1e-9),
                             writer_backpressure_seconds=self.backpressure_seconds)
        if self.error:
            self.metadata['error'] = repr(self.error)
        self._write_metadata()
        if self.error is None and not aborted:
            with h5py.File(self.path/'episode.partial.h5', 'a') as file:
                file.attrs['metadata_json'] = json.dumps(self.metadata)
                file.attrs['complete'] = True
            (self.path/'episode.partial.h5').replace(self.path/'episode.h5')
        self.check()
        return self.path
