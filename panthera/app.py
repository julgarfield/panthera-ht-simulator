from pathlib import Path
import argparse
import hashlib
import json
import time
import traceback
import glfw
import mujoco
import numpy as np
from PIL import Image
from .config import load_config, resolve, ROOT
from .environment import SimulatedPanthera
from .interfaces import Action
from .keyboard import KeyboardController
from .recorder import EpisodeRecorder
from .rendering import Renderer
from .replay import EpisodeReplay, export_episode


def scripted_action(t):
    """Small deterministic test motions, never used during normal keyboard control."""
    velocity = np.zeros(6)
    phase = int(t/.6)
    if phase < 12:
        velocity[phase//2] = .10 if phase % 2 == 0 else -.10
        return Action(joint_velocity=velocity, gripper_velocity=-.015 if t < 2 else .015 if t < 4 else 0)
    if t < 8:
        return Action(mode='cartesian', cartesian_twist=np.array([.025, 0, .025, 0, 0, 0]))
    return Action()


def run_live(args):
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.setdefault('task', {})['seed'] = args.seed
    env = SimulatedPanthera(cfg)
    renderer = Renderer(env, visible=not args.headless)
    keyboard = KeyboardController(renderer.window, cfg)
    print(f'Robot loaded: official Panthera-HT, 6 axes + gripper. Renderer: {renderer.gpu}', flush=True)
    recorder = None
    pending_record = args.record
    paused = False
    message = 'Ready. SPACE starts a new episode.'
    stride = cfg['simulation']['control_hz']//cfg['table_camera']['fps']
    clock_start = next_control = next_display = report_time = time.perf_counter()
    last_tick = last_frames = frames = displays = 0
    total_ticks = 0
    camera_ms = []
    rates = dict(physics_fps=0, camera_fps=0, realtime_factor=0)
    completed_episodes = []
    failure = False
    try:
        while not glfw.window_should_close(renderer.window):
            glfw.poll_events()
            now = time.perf_counter()
            while keyboard.events:
                event = keyboard.events.popleft()
                if event == 'quit':
                    glfw.set_window_should_close(renderer.window, True)
                elif event in ('pause', 'focus_lost'):
                    paused = not paused if event == 'pause' else True
                    keyboard.held.clear()
                    keyboard.homing = False
                    env.hold()
                    next_control = now
                elif event == 'hold':
                    env.hold()
                elif event == 'record':
                    pending_record = not pending_record
                    message = 'Recording toggle queued for next synchronized camera frame.'
                elif event == 'home':
                    keyboard.homing = True
                    message = 'Moving to configured simulation work pose.'
                elif event == 'reset':
                    if recorder:
                        completed_episodes.append(str(recorder.close()))
                        recorder = None
                    env.reset()
                    keyboard.held.clear()
                    keyboard.homing = False
                    pending_record = False
                    last_tick = 0
                    message = 'Scene reset; previous episode saved.'
            if recorder:
                recorder.check()
            if not paused and now >= next_control:
                capture_due = env.tick % stride == 0
                if pending_record and capture_due:
                    if recorder:
                        path = recorder.close()
                        completed_episodes.append(str(path))
                        message = f'Saved {path.name}: {env.task.metadata()["outcome"].upper()}'
                        print(f'Episode saved: {path}', flush=True)
                        recorder = None
                    else:
                        recorder = EpisodeRecorder(cfg, env, renderer.gpu, args.dataset_directory)
                        message = 'Recording states, commands and both cameras.'
                    pending_record = False
                observation = env.get_observation()
                action = scripted_action(env.timestamp) if args.scripted else keyboard.get_action(observation)
                target = env.apply_action(action)
                if recorder:
                    recorder.record_control(observation, action, target)
                if capture_due:
                    images, poses = renderer.capture()
                    frames += 1
                    camera_ms.append(renderer.capture_ms)
                    if recorder:
                        recorder.record_frame(observation, action, target, images, poses)
                env.advance()
                total_ticks += 1
                next_control += env.dt
                # Preserve every simulation tick and every camera pair under overload.
                # Slow simulation time rather than skipping data or creating a catch-up spiral.
                if now-next_control > .25:
                    next_control = now
                    message = 'Running below real time. Lower camera resolution or choose the NVIDIA GPU.'
            elif paused:
                next_control = now
            if now-report_time >= 1:
                elapsed = now-report_time
                rates = dict(physics_fps=(env.tick-last_tick)*env.substeps/elapsed,
                             camera_fps=(frames-last_frames)/elapsed,
                             realtime_factor=(env.tick-last_tick)*env.dt/elapsed)
                last_tick, last_frames, report_time = env.tick, frames, now
            if now >= next_display:
                renderer.draw(dict(mode='home' if keyboard.homing else keyboard.mode, selected=keyboard.selected,
                                   speed=keyboard.speed, paused=paused, recording=recorder is not None,
                                   episode=recorder.path.name if recorder else '', message=message, **rates))
                displays += 1
                next_display = now+1/cfg['simulation']['display_hz']
            if args.seconds and env.timestamp >= args.seconds-1e-9:
                break
            if not args.unthrottled:
                delay = min(next_control if not paused else now+.004, next_display)-time.perf_counter()
                if delay > 0:
                    time.sleep(min(delay, .002))
            else:
                next_control = time.perf_counter()
        if args.snapshot:
            folder = Path(args.snapshot)
            folder.mkdir(parents=True, exist_ok=True)
            for name, pixels in renderer.images.items():
                Image.fromarray(pixels).save(folder/f'{name}.png')
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, renderer.context)
            width, height = glfw.get_framebuffer_size(renderer.window)
            pixels = np.empty((height, width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(pixels, None, mujoco.MjrRect(0, 0, width, height), renderer.context)
            Image.fromarray(pixels[::-1]).save(folder/'application.png')
    except BaseException:
        failure = True
        raise
    finally:
        try:
            if recorder:
                path = recorder.close(aborted=failure)
                completed_episodes.append(str(path))
                print(f'Episode saved: {path}', flush=True)
        finally:
            renderer.close()
    wall = time.perf_counter()-clock_start
    report = {'gpu': renderer.gpu, 'simulation_seconds': env.timestamp, 'wall_seconds': wall,
              'camera_pairs': frames, 'camera_pairs_per_wall_second': frames/wall,
              'physics_steps_per_wall_second': total_ticks*env.substeps/wall,
              'display_frames': displays, 'camera_pair_median_ms': float(np.median(camera_ms)) if camera_ms else 0,
              'camera_pair_p95_ms': float(np.percentile(camera_ms, 95)) if camera_ms else 0,
              'limit_guard_corrections': env.limit_corrections, 'episodes': completed_episodes}
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')


def run_replay(args):
    episode = EpisodeReplay(args.replay)
    renderer = None
    try:
        cfg = episode.config()
        current_hash = hashlib.sha256(resolve(cfg, cfg['robot']['urdf_path']).read_bytes()).hexdigest()
        if current_hash != episode.metadata['urdf_sha256']:
            raise ValueError('Official URDF differs from the recorded episode; restore matching assets')
        env = SimulatedPanthera(cfg)
        if 'task' not in episode.metadata:
            env.model.site_rgba[env.model.site('placement_target').id, 3] = 0
        else:
            env.model.site_pos[env.model.site('placement_target').id] = episode.metadata['task']['target_center']
        renderer = Renderer(env, visible=not args.headless)
        keyboard = KeyboardController(renderer.window, cfg)
        start = time.perf_counter()
        pause_start = None
        elapsed = 0.0
        while not glfw.window_should_close(renderer.window):
            glfw.poll_events()
            while keyboard.events:
                event = keyboard.events.popleft()
                if event == 'quit':
                    glfw.set_window_should_close(renderer.window, True)
                elif event in ('pause', 'focus_lost'):
                    if pause_start is None:
                        pause_start = time.perf_counter()
                    elif event == 'pause':
                        start += time.perf_counter()-pause_start
                        pause_start = None
            elapsed = ((pause_start or time.perf_counter())-start)*args.speed
            i = int(np.clip(np.searchsorted(episode.control_times, elapsed, side='right')-1, 0, len(episode.control_times)-1))
            env.set_replay_state(episode.qpos[i], episode.qvel[i], float(episode.file['control/simulation_timestamps'][i]))
            if args.live_replay_cameras:
                renderer.capture()
                images = None
            else:
                images = episode.images(elapsed)
            renderer.draw(dict(mode='REPLAY', paused=pause_start is not None, replay=not args.live_replay_cameras,
                               message=f'{episode.path.name} | {elapsed:.2f}s / {episode.metadata["episode_duration"]:.2f}s | P pause | ESC exit'), images)
            if elapsed >= episode.metadata['episode_duration']:
                if args.loop:
                    start = time.perf_counter()
                else:
                    break
            time.sleep(1/cfg['simulation']['display_hz'])
        print(f'Replay complete: {episode.path.name}', flush=True)
    finally:
        if renderer:
            renderer.close()
        episode.close()


def main():
    parser = argparse.ArgumentParser(description='Official Panthera-HT keyboard simulator and dataset recorder')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--replay', type=Path, help='Episode folder or episode.h5')
    parser.add_argument('--export', type=Path, help='Export episode to CSV and PNG')
    parser.add_argument('--validate', type=Path, help='Validate episode shapes, timestamps and alignment')
    parser.add_argument('--output', type=Path, help='Export output folder')
    parser.add_argument('--record', action='store_true', help='Record immediately at first camera tick')
    parser.add_argument('--seed', type=int, help='Reproduce a task layout; each reset increments the seed')
    parser.add_argument('--dataset-directory', type=Path)
    parser.add_argument('--headless', action='store_true', help='Hidden OpenGL window (display driver still required)')
    parser.add_argument('--seconds', type=float, help='Stop after this many simulation seconds')
    parser.add_argument('--scripted', action='store_true', help='Small scripted test motions instead of keyboard input')
    parser.add_argument('--unthrottled', action='store_true', help='Run as fast as possible; do not target wall-clock time')
    parser.add_argument('--report', type=Path, help='Write performance report JSON')
    parser.add_argument('--snapshot', type=Path, help='Save final camera and application images')
    parser.add_argument('--speed', type=float, default=1.0, help='Replay playback speed')
    parser.add_argument('--loop', action='store_true', help='Loop replay')
    parser.add_argument('--live-replay-cameras', action='store_true', help='Render cameras during replay instead of showing recorded RGB')
    args = parser.parse_args()
    if args.seed is not None and args.seed < 0:
        parser.error('Seed must be non-negative')
    if args.speed <= 0 or (args.seconds is not None and args.seconds <= 0):
        parser.error('Speed and seconds must be positive')
    if args.headless and not (args.seconds or args.replay or args.export or args.validate):
        parser.error('--headless requires --seconds or an episode operation')
    if args.export:
        print(f'Exported: {export_episode(args.export, args.output)}')
    elif args.validate:
        replay = EpisodeReplay(args.validate)
        print(f'Valid episode: {len(replay.frame_times)} synchronized camera pairs; {len(replay.control_times)} control samples')
        replay.close()
    elif args.replay:
        run_replay(args)
    else:
        run_live(args)


if __name__ == '__main__':
    main()
