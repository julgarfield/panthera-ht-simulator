import mujoco
import numpy as np
import pytest
from panthera.interfaces import Action
from panthera.policy import PolicyController, pack_targets


def test_seed_reproduces_layout_and_reset_changes_it(env):
    env.reset(seed=127)
    layout = env.data.qpos.copy(), env.task.target.copy()
    env.step(Action(joint_velocity=np.ones(6)*.1))
    env.reset(seed=127)
    np.testing.assert_array_equal(env.data.qpos, layout[0])
    np.testing.assert_array_equal(env.task.target, layout[1])
    env.reset()
    assert env.task.seed == 128
    assert not np.array_equal(env.task.target, layout[1])
    assert not env.task.success and not env.task.lifted
    with pytest.raises(ValueError):
        env.reset(seed=-1)


def test_success_requires_lift_release_containment_dwell_and_current_placement(env):
    qid = env.model.jnt_qposadr[env.task.cube_joint]
    vid = env.model.jnt_dofadr[env.task.cube_joint]
    def place(xy, rotation=(1, 0, 0, 0)):
        env.data.qpos[qid:qid+7] = np.r_[xy, env.task.table_z+env.task.half_size[2]-.0001, rotation]
        env.data.qvel[vid:vid+6] = 0
        mujoco.mj_forward(env.model, env.data)
    place(env.task.target[:2])
    for _ in range(80):
        env.task.update()
    assert not env.task.success  # simply spawning/sliding into the target is insufficient
    env.task.lifted = True  # isolate placement predicate; actual grasp/lift covered separately
    for _ in range(59):
        env.task.update()
    assert not env.task.success
    env.task.update()
    assert env.task.success
    # Center inside is insufficient if a cube corner lies outside.
    place(env.task.target[:2] + [.045, 0])
    env.task.update()
    assert not env.task.success
    place(env.task.target[:2])
    env.data.qvel[vid] = .1
    env.task.update()
    assert env.task.settled_ticks == 0
    env.task.begin_episode()
    assert not env.task.lifted and not env.task.success


def test_policy_preserves_all_four_commands_and_physics(env):
    blocks = []
    # Fast changes within the camera interval must survive conversion exactly.
    for k in range(40):
        targets = []
        for sub in range(4):
            v = np.zeros(6)
            v[0] = [.12, -.08, .04, 0][sub]
            env.apply_action(Action(joint_velocity=v, gripper_velocity=-.005))
            targets.append(env.targets.copy())
            env.advance()
        blocks.append(pack_targets(targets))
    expected_qpos = env.data.qpos.copy()
    env.reset(seed=0)
    controller = PolicyController(env)
    for block in blocks:
        _, executed = controller.step(block)
        np.testing.assert_allclose(executed[:, :7], block.reshape(4, 7), atol=1e-7, rtol=0)
        np.testing.assert_array_equal(executed[:, 7], -executed[:, 6])
    assert env.tick == 160
    np.testing.assert_allclose(env.data.qpos[env.qids], expected_qpos[env.qids], atol=2e-6, rtol=0)
    before = env.data.qpos.copy()
    for invalid in (np.zeros(7), np.full(28, np.nan)):
        with pytest.raises(ValueError):
            controller.step(invalid)
        np.testing.assert_array_equal(env.data.qpos, before)
    _, guarded = controller.step(np.full(28, 1e6))
    assert np.all(guarded >= env.lower) and np.all(guarded <= env.upper)
