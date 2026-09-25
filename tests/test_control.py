import glfw
import mujoco
import numpy as np
import pytest
from collections import deque
from panthera.interfaces import Action
from panthera.keyboard import KeyboardController


@pytest.mark.parametrize('joint', range(6))
def test_each_joint_moves_and_release_holds(env, joint):
    initial = env.get_observation()['joint_positions'].copy()
    v = np.zeros(6)
    v[joint] = .12
    for _ in range(120):
        env.step(Action(joint_velocity=v))
    assert env.get_observation()['joint_positions'][joint]-initial[joint] > .07
    env.step(Action())
    stopped = env.targets[:6].copy()
    for _ in range(120):
        env.step(Action())
    np.testing.assert_allclose(env.get_observation()['joint_positions'], stopped, atol=.004)
    assert np.isfinite(env.data.qpos).all()


def test_huge_and_nonfinite_commands(env):
    for _ in range(240):
        env.step(Action(joint_velocity=np.array([1e6,1e6,1e6,-1e6,1e6,1e6]), gripper_velocity=-100))
        assert np.all(env.data.qpos[env.qids] >= env.lower)
        assert np.all(env.data.qpos[env.qids] <= env.upper)
        assert np.all(np.abs(env.data.actuator_force[env.aids]) <= env.max_effort+1e-10)
    state = env.data.qpos.copy()
    with pytest.raises(ValueError):
        env.step(Action(joint_velocity=np.full(6, np.nan)))
    np.testing.assert_array_equal(state, env.data.qpos)


def test_cartesian_translation_and_wrist_mount(env):
    initial = env.get_observation()['end_effector_pose'][:3]
    camera_id = env.model.camera('wrist_camera').id
    original_camera = env.data.cam_xpos[camera_id].copy()
    for _ in range(120):
        env.step(Action(mode='cartesian', cartesian_twist=np.array([0,.03,-.03,0,0,0])))
    delta = env.get_observation()['end_effector_pose'][:3]-initial
    np.testing.assert_allclose(delta, [0,.03,-.03], atol=.01)
    assert np.linalg.norm(env.data.cam_xpos[camera_id]-original_camera) > .02
    parent = env.model.body(env.cfg['wrist_camera']['parent_link']).id
    local = env.data.xmat[parent].reshape(3,3).T @ (env.data.cam_xpos[camera_id]-env.data.xpos[parent])
    np.testing.assert_allclose(local, env.cfg['wrist_camera']['position_offset'], atol=1e-12)


def test_grasp_and_lift_with_contacts(env):
    initial_height = env.get_observation()['object_poses'][0,2]
    for goal, grip, seconds in [([.34,.02,.764], 0, 4), ([.34,.02,.764], -.025, 2), ([.34,.02,.94], 0, 4)]:
        for _ in range(int(seconds/env.dt)):
            error = np.asarray(goal)-env.get_observation()['end_effector_pose'][:3]
            env.step(Action(mode='cartesian', cartesian_twist=np.r_[np.clip(error*2,-.06,.06),0,0,0], gripper_velocity=grip))
    assert env.get_observation()['object_poses'][0,2] > initial_height+.10
    assert env.targets[6] < env.data.qpos[env.qids[6]]  # contact pressure remains on key release


def test_keyboard_press_hold_release_and_focus(env):
    keyboard = KeyboardController.__new__(KeyboardController)
    keyboard.cfg, keyboard.held, keyboard.events = env.cfg, set(), deque()
    keyboard.mode, keyboard.selected, keyboard.speed, keyboard.homing = 'joint', 0, 1.0, False
    keyboard.on_key(None, glfw.KEY_3, 0, glfw.PRESS, 0)
    keyboard.on_key(None, glfw.KEY_E, 0, glfw.PRESS, 0)
    assert keyboard.get_action(env.get_observation()).joint_velocity[2] > 0
    assert keyboard.get_action(env.get_observation()).joint_velocity[2] > 0
    keyboard.on_key(None, glfw.KEY_E, 0, glfw.RELEASE, 0)
    assert not keyboard.get_action(env.get_observation()).joint_velocity.any()
    keyboard.on_key(None, glfw.KEY_E, 0, glfw.PRESS, 0)
    keyboard.on_focus(None, False)
    assert not keyboard.held
    assert keyboard.events.pop() == 'focus_lost'
