import json
import h5py
import numpy as np
import pytest
from PIL import Image
from panthera.interfaces import Action
from panthera.recorder import EpisodeRecorder
from panthera.rendering import Renderer
from panthera.replay import EpisodeReplay, export_episode


def test_rgb_record_export_and_replay(env, tmp_path):
    renderer = Renderer(env, visible=False)
    recorder = EpisodeRecorder(env.cfg, env, renderer.gpu, tmp_path)
    try:
        for tick in range(120):
            action = Action(joint_velocity=np.array([.06,0,0,0,0,0]))
            observation = env.get_observation()
            target = env.apply_action(action)
            recorder.record_control(observation, action, target)
            if tick % 4 == 0:
                images, poses = renderer.capture()
                recorder.record_frame(observation, action, target, images, poses)
            env.advance()
        path = recorder.close()
        replay = EpisodeReplay(path)
        try:
            assert len(replay.control_times) == 120
            assert len(replay.frame_times) == 30
            for i in (0, 37, 119):
                env.set_replay_state(replay.qpos[i], replay.qvel[i], replay.control_times[i])
                np.testing.assert_array_equal(env.data.qpos, replay.qpos[i])
                np.testing.assert_allclose(env.get_observation()['end_effector_pose'], replay.file['control/end_effector_pose'][i], atol=1e-12)
            for name in ('table_camera', 'wrist_camera'):
                ds = replay.file[f'observations/images/{name}']
                assert ds.shape == (30,480,640,3)
                assert ds[0].std() > 10
                assert not np.array_equal(ds[0], ds[-1])
            env.set_replay_state(replay.qpos[0], replay.qvel[0], 0)
            rerendered, _ = renderer.capture()
            for name in ('table_camera', 'wrist_camera'):
                np.testing.assert_array_equal(rerendered[name], replay.file[f'observations/images/{name}'][0])
        finally:
            replay.close()
        exported = export_episode(path)
        with h5py.File(path/'episode.h5') as f:
            np.testing.assert_array_equal(np.array(Image.open(exported/'table_camera/000010.png')), f['observations/images/table_camera'][10])
        assert len((exported/'control.csv').read_text().splitlines()) == 121
        with pytest.raises(FileExistsError):
            export_episode(path)
        next_episode = EpisodeRecorder(env.cfg, env, renderer.gpu, tmp_path)
        assert next_episode.path.name == 'episode_0002'
        next_episode.close(aborted=True)
        assert json.loads((next_episode.path/'metadata.json').read_text())['status'] == 'incomplete'
        with pytest.raises(ValueError):
            EpisodeReplay(next_episode.path)
    finally:
        if not recorder.closed:
            recorder.close(aborted=True)
        renderer.close()
