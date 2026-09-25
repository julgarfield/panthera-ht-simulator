"""Optional integration suite: run in .venv-export; no model/tokenizer downloads."""
import json
from importlib.util import find_spec
import h5py
import numpy as np
import pytest
from panthera.interfaces import Action
from panthera.recorder import EpisodeRecorder
from panthera.rendering import Renderer
from panthera.lerobot_export import export_dataset, validate_dataset
from panthera.policy import PolicyController

pytestmark = pytest.mark.skipif(find_spec('lerobot') is None, reason='Optional export environment required')


def record_short(env, renderer, root, seed, ticks=60):
    env.reset(seed=seed)
    recorder = EpisodeRecorder(env.cfg, env, renderer.gpu, root)
    try:
        for tick in range(ticks):
            action = Action(joint_velocity=np.array([.1 if tick % 4 < 2 else -.07, 0, 0, 0, 0, 0]))
            obs = env.get_observation()
            target = env.apply_action(action)
            recorder.record_control(obs, action, target)
            if tick % 4 == 0:
                images, poses = renderer.capture()
                recorder.record_frame(obs, action, target, images, poses)
            env.advance()
    finally:
        recorder.close()
    return recorder.path


def test_real_lerobot_roundtrip_two_episodes_and_pi05_features(env, tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HOME', str(tmp_path/'hf-cache'))
    renderer = Renderer(env, visible=False)
    try:
        paths = [record_short(env, renderer, tmp_path/'source', seed) for seed in (10, 11)]
        with pytest.raises(ValueError, match='No eligible episodes'):
            export_dataset(tmp_path/'source', tmp_path/'excluded')
        result = export_dataset(tmp_path/'source', tmp_path/'dataset', include_failures=True)
        assert result['frames'] == 30 and result['episodes'] == 2
        assert result['preserved_control_samples'] == 120
        assert validate_dataset(tmp_path/'dataset', compare_sources=True)['valid']
        from lerobot.datasets import LeRobotDataset
        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        ds = LeRobotDataset('local/panthera-red-cube', root=tmp_path/'dataset', video_backend='pyav',
                            delta_timestamps={'action': [k/30 for k in range(5)]})
        # Near the end of episode 0 the sequence must pad, never draw from episode 1.
        row = ds[14]
        assert row['action'].shape == (5, 28)
        assert row['action_is_pad'].tolist() == [False, True, True, True, True]
        np.testing.assert_array_equal(row['action'][0], row['action'][4])
        config = PI05Config(device='cpu', input_features={
            'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)) for key in ds.meta.camera_keys}},
            output_features={'action': PolicyFeature(type=FeatureType.ACTION, shape=(28,))})
        config.validate_features()
        assert len(config.image_features) == 2
        assert config.max_action_dim >= 28 and not config.use_relative_actions
        # Real exported row -> native simulator, including all subframe changes.
        env.reset(seed=10)
        controller = PolicyController(env)
        obs = controller.observe(renderer)
        assert obs['observation.state'].shape == (7,)
        _, targets = controller.step(ds[0]['action'][0].numpy())
        with h5py.File(paths[0]/'episode.h5') as source:
            np.testing.assert_allclose(targets, source['control/joint_position_target'][:4], atol=1e-7, rtol=0)
        assert env.tick == 4
        with pytest.raises(FileExistsError):
            export_dataset(tmp_path/'source', tmp_path/'dataset', include_failures=True)
    finally:
        renderer.close()


def test_truncated_block_rejected_and_hdf5_retained(env, tmp_path, monkeypatch):
    monkeypatch.setenv('HF_HOME', str(tmp_path/'hf-cache'))
    renderer = Renderer(env, visible=False)
    try:
        path = record_short(env, renderer, tmp_path/'source', 13, ticks=7)
        before = (path/'episode.h5').read_bytes()
        with pytest.raises(ValueError, match='Incomplete final'):
            export_dataset(path, tmp_path/'bad', include_failures=True)
        assert before == (path/'episode.h5').read_bytes()
        assert not (tmp_path/'bad').exists()
        assert json.loads((path/'metadata.json').read_text())['task']['outcome'] == 'failure'
    finally:
        renderer.close()
