"""Offline, lossless HDF5 -> LeRobot export. Optional dependencies only here."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import numpy as np
from .policy import ACTION_CONTRACT, ACTION_NAMES, STATE_NAMES, pack_targets, policy_state
from .replay import EpisodeReplay

CAMERAS = ('table_camera', 'wrist_camera')
LEROBOT_VERSION = '0.6.1'


def lerobot_class():
    # All operations are local. Missing data must fail instead of fetching a Hub repo.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    try:
        if version('lerobot') != LEROBOT_VERSION:
            raise RuntimeError(f'Exporter is validated with lerobot=={LEROBOT_VERSION}; use requirements-export.txt')
        from lerobot.datasets import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError('Install requirements-export.txt into a separate export environment') from exc
    return LeRobotDataset


def discover(source):
    source = Path(source).resolve()
    if source.is_file():
        if source.name != 'episode.h5':
            raise ValueError('Only completed episode.h5 files can be exported')
        return [source.parent]
    if (source/'metadata.json').exists():
        return [source]
    paths = sorted(p.parent for p in source.glob('episode_*/metadata.json'))
    if not paths:
        raise ValueError(f'No HDF5 episodes found under {source}')
    return paths


def check_source(episode):
    """Strict contract preflight; no command decimation, padding or silent trim."""
    m, f = episode.metadata, episode.file
    if not bool(f.attrs.get('complete', False)) or json.loads(f.attrs['metadata_json']) != m:
        raise ValueError('HDF5 completion/metadata does not match metadata.json')
    if m.get('action_contract') != ACTION_CONTRACT or m['schema_version'] != '1.1':
        raise ValueError('Legacy episode has no verified task/action contract; keep it for review')
    task = m.get('task', {})
    if not isinstance(task.get('instruction'), str) or not task['instruction'].strip():
        raise ValueError('Missing task instruction')
    if type(task.get('seed')) is not int or not 0 <= task['seed'] <= np.iinfo(np.int64).max:
        raise ValueError('Missing or invalid seed')
    if type(task.get('success')) is not bool or task.get('outcome') != ('success' if task['success'] else 'failure'):
        raise ValueError('Missing or inconsistent success label')
    if m['control_hz'] != 120 or m['camera_fps'] != 30 or m['configuration']['wrist_camera']['fps'] != 30:
        raise ValueError('The 4x7 action contract requires 120 Hz control and 30 Hz cameras')
    if m['joint_names'] != STATE_NAMES[:6] or m['gripper_joint_names'] != ['gripper_L_joint', 'gripper_R_joint']:
        raise ValueError('Unexpected joint order for this action contract')
    n = len(episode.frame_times)
    if len(episode.control_times) != 4*n:
        raise ValueError('Incomplete final 4-control block; stop recording with Space on a camera boundary')
    np.testing.assert_array_equal(f['control_index'][:], np.arange(n)*4)
    np.testing.assert_array_equal(f['frame_index'][:], np.arange(n))
    np.testing.assert_array_equal(f['control/index'][:], np.arange(4*n))
    np.testing.assert_allclose(episode.frame_times, np.arange(n)/30, rtol=0, atol=1e-9)
    np.testing.assert_allclose(episode.control_times, np.arange(4*n)/120, rtol=0, atol=1e-9)
    for prefix, times in (('', episode.frame_times), ('control/', episode.control_times)):
        np.testing.assert_allclose(f[f'{prefix}simulation_timestamps'][:], times+m['start_simulation_time'], rtol=0, atol=1e-9)
    for group in ('control', 'observations'):
        for name, width in (('joint_positions', 6), ('gripper_positions', 2)):
            values = f[f'{group}/{name}'][:]
            if values.shape != (4*n if group == 'control' else n, width) or not np.isfinite(values).all():
                raise ValueError(f'Invalid {group}/{name}')
    for i in range(n):
        pack_targets(f['control/joint_position_target'][4*i:4*i+4])


def features_for(metadata):
    features = {
        'observation.state': dict(dtype='float32', shape=(7,), names=STATE_NAMES),
        'action': dict(dtype='float32', shape=(28,), names=ACTION_NAMES),
        'source.control_timestamps': dict(dtype='float64', shape=(4,), names=[f'tick{i}' for i in range(4)]),
        'source.control_index': dict(dtype='int64', shape=(4,), names=[f'tick{i}' for i in range(4)]),
        'source.seed': dict(dtype='int64', shape=(1,), names=None),
        'next.done': dict(dtype='bool', shape=(1,), names=None),
        'next.success': dict(dtype='bool', shape=(1,), names=None),
    }
    for name in CAMERAS:
        c = metadata['configuration'][name]
        features[f'observation.images.{name}'] = dict(dtype='image', shape=(c['height'], c['width'], 3), names=['height', 'width', 'channels'])
    return features


def frame_from_source(episode, i):
    f, m = episode.file, episode.metadata
    last = i == len(episode.frame_times)-1
    obs = {key: f[f'observations/{key}'][i] for key in ('joint_positions', 'gripper_positions')}
    return {
        'observation.state': policy_state(obs),
        'action': pack_targets(f['control/joint_position_target'][4*i:4*i+4]),
        'source.control_timestamps': f['control/timestamps'][4*i:4*i+4],
        'source.control_index': np.arange(4*i, 4*i+4, dtype=np.int64),
        'source.seed': np.array([m['task']['seed']], dtype=np.int64),
        'next.done': np.array([last], dtype=bool),
        'next.success': np.array([last and m['task']['success']], dtype=bool),
        'task': m['task']['instruction'],
        **{f'observation.images.{name}': f[f'observations/images/{name}'][i] for name in CAMERAS},
    }


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_dataset(source, destination, repo_id='local/panthera-red-cube', include_failures=False):
    destination = Path(destination).resolve()
    staging = destination.with_name(destination.name+'.partial')
    if destination.exists() or staging.exists():
        raise FileExistsError(f'Output or partial output already exists: {destination}')
    selected, skipped, expected_features = [], [], None
    # Preflight everything before creating a dataset. A malformed selected episode
    # aborts the export rather than silently shrinking the training set.
    for path in discover(source):
        m = json.loads((path/'metadata.json').read_text(encoding='utf-8'))
        reason = None
        if m.get('status') != 'complete':
            reason = 'incomplete recording'
        elif m.get('schema_version') != '1.1' or 'task' not in m:
            reason = 'legacy recording without verified task labels'
        elif not include_failures and m['task'].get('success') is False:
            reason = 'failed trial (excluded by default)'
        if reason:
            skipped.append(dict(source=str(path), reason=reason))
            continue
        episode = EpisodeReplay(path)
        try:
            check_source(episode)
            features = features_for(m)
            if expected_features is not None and features != expected_features:
                raise ValueError('Selected episodes have incompatible camera shapes')
            expected_features = features
            selected.append(dict(source=str(path), frames=len(episode.frame_times), task=m['task'],
                                 configuration=m['configuration'], urdf_sha256=m['urdf_sha256'],
                                 hdf5_sha256=file_hash(path/'episode.h5')))
        finally:
            episode.close()
    if not selected:
        raise ValueError(f'No eligible episodes. Exclusions: {json.dumps(skipped)}')
    dataset = lerobot_class().create(repo_id=repo_id, root=staging, fps=30,
                                    robot_type='panthera_ht_6dof', features=expected_features,
                                    use_videos=False, image_writer_threads=2, video_backend='pyav')
    try:
        for item in selected:
            episode = EpisodeReplay(item['source'])
            try:
                for i in range(item['frames']):
                    dataset.add_frame(frame_from_source(episode, i))
                dataset.save_episode()
            finally:
                episode.close()
    finally:
        dataset.finalize()
    manifest = dict(repo_id=repo_id, lerobot_version=LEROBOT_VERSION, action_contract=ACTION_CONTRACT,
                    fps=30, control_hz=120, storage='lossless image features',
                    include_failures=include_failures, episodes=selected, skipped=skipped)
    (staging/'panthera_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    del dataset
    report = validate_dataset(staging, compare_sources=True)
    (staging/'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    staging.rename(destination)
    # Also reload after relocation: no hidden dependency on staging image paths.
    validate_dataset(destination)
    return report


def validate_dataset(root, compare_sources=False):
    root = Path(root).resolve()
    manifest = json.loads((root/'panthera_manifest.json').read_text(encoding='utf-8'))
    dataset = lerobot_class()(repo_id=manifest['repo_id'], root=root, video_backend='pyav')
    expected_count = sum(item['frames'] for item in manifest['episodes'])
    if len(dataset) != expected_count or dataset.num_episodes != len(manifest['episodes']):
        raise ValueError('Exported episode/frame counts disagree')
    if dataset.fps != 30 or set(dataset.meta.camera_keys) != {f'observation.images.{n}' for n in CAMERAS}:
        raise ValueError('Exported FPS/camera features disagree')
    for key, shape in (('observation.state', (7,)), ('action', (28,))):
        if tuple(dataset.features[key]['shape']) != shape:
            raise ValueError(f'Invalid exported {key} dimensions')
        for stat in ('min', 'max', 'mean', 'std', 'q01', 'q99'):
            if stat not in dataset.meta.stats[key] or not np.isfinite(dataset.meta.stats[key][stat]).all():
                raise ValueError(f'Missing/invalid {key} {stat} normalization statistics')
    offset = 0
    for e, item in enumerate(manifest['episodes']):
        episode = EpisodeReplay(item['source']) if compare_sources else None
        try:
            if episode and file_hash(episode.path/'episode.h5') != item['hdf5_sha256']:
                raise ValueError('Source HDF5 changed after export')
            meta = dataset.meta.episodes[e]
            if int(meta['dataset_from_index']) != offset or int(meta['dataset_to_index']) != offset+item['frames']:
                raise ValueError('Episode boundary metadata mismatch')
            for i in range(item['frames']):
                row = dataset[offset+i]
                if int(row['episode_index']) != e or int(row['frame_index']) != i or int(row['index']) != offset+i:
                    raise ValueError('Episode/frame boundary mismatch')
                if abs(float(row['timestamp'])-i/30) > 1e-4 or row['task'] != item['task']['instruction']:
                    raise ValueError('Timestamp/task mismatch')
                if int(row['source.seed']) != item['task']['seed']:
                    raise ValueError('Seed mismatch')
                if bool(row['next.done']) != (i == item['frames']-1) or bool(row['next.success']) != (i == item['frames']-1 and item['task']['success']):
                    raise ValueError('Terminal label mismatch')
                np.testing.assert_array_equal(row['source.control_index'].numpy(), 4*i+np.arange(4))
                # LeRobot's tensor transform casts floating features to float32.
                np.testing.assert_array_equal(row['source.control_timestamps'].numpy(), ((4*i+np.arange(4))/120).astype(np.float32))
                for key, shape in (('observation.state', (7,)), ('action', (28,))):
                    if tuple(row[key].shape) != shape or not np.isfinite(row[key].numpy()).all():
                        raise ValueError(f'Invalid loaded {key}')
                expected = frame_from_source(episode, i) if episode else None
                if expected:
                    for key in ('observation.state', 'action'):
                        np.testing.assert_array_equal(row[key].numpy(), expected[key])
                for name in CAMERAS:
                    key = f'observation.images.{name}'
                    rgb = row[key].numpy()
                    h, w, _ = dataset.features[key]['shape']
                    if rgb.shape != (3, h, w) or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                        raise ValueError(f'Invalid decoded RGB: {key}')
                    if expected:
                        restored = np.rint(rgb.transpose(1, 2, 0)*255).astype(np.uint8)
                        np.testing.assert_array_equal(restored, expected[key])
            offset += item['frames']
        finally:
            if episode:
                episode.close()
    return dict(valid=True, episodes=dataset.num_episodes, frames=len(dataset),
                camera_frames={name: len(dataset) for name in CAMERAS}, state_shape=[7], action_shape=[28],
                fps=30, preserved_control_samples=4*len(dataset), compared_to_hdf5=compare_sources,
                skipped=manifest['skipped'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Episode folder, episode.h5, or recordings directory')
    parser.add_argument('--output', type=Path, help='New local LeRobot dataset directory')
    parser.add_argument('--repo-id', default='local/panthera-red-cube', help='Local dataset identifier; never uploaded')
    parser.add_argument('--include-failures', action='store_true', help='Include failed trials for review only')
    parser.add_argument('--validate', type=Path, help='Reload and validate an existing export')
    parser.add_argument('--compare-sources', action='store_true', help='Also compare every RGB pixel/state/action to original HDF5')
    args = parser.parse_args()
    if args.validate:
        report = validate_dataset(args.validate, args.compare_sources)
    elif args.source and args.output:
        report = export_dataset(args.source, args.output, args.repo_id, args.include_failures)
    else:
        parser.error('Use --source and --output, or --validate')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
