"""One OpenGL context renders the viewport and both synchronized RGB sensors."""
import time
import glfw
import mujoco
import numpy as np
from PIL import Image
from OpenGL import GL


class Renderer:
    def __init__(self, env, visible=True):
        self.env, self.cfg = env, env.cfg
        self.visible = visible
        if not glfw.init():
            raise RuntimeError('GLFW initialization failed. Install/update the GPU display driver.')
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
        self.window = glfw.create_window(self.cfg['ui']['width'], self.cfg['ui']['height'], 'Panthera-HT | Teleoperation & demonstrations', None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('Could not create OpenGL window. See README graphics troubleshooting.')
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        self.gpu = GL.glGetString(GL.GL_RENDERER).decode()
        self.context = mujoco.MjrContext(env.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        self.scene = mujoco.MjvScene(env.model, maxgeom=2000)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = self.cfg['ui']['camera_distance']
        self.camera.azimuth = self.cfg['ui']['camera_azimuth']
        self.camera.elevation = self.cfg['ui']['camera_elevation']
        self.camera.lookat[:] = self.cfg['ui']['camera_lookat']
        self.sensor_camera = mujoco.MjvCamera()
        self.sensor_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.images = {}
        self.capture_ms = 0.0
        self.resized_cache = {}
        self.last_cursor = None
        glfw.set_cursor_pos_callback(self.window, self._mouse)
        glfw.set_scroll_callback(self.window, self._scroll)

    def _mouse(self, window, x, y):
        previous = self.last_cursor
        self.last_cursor = (x, y)
        if previous is None:
            return
        width, height = glfw.get_window_size(window)
        if x > width*.66:
            return
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            mode = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            mode = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            return
        mujoco.mjv_moveCamera(self.env.model, mode, (x-previous[0])/height, (y-previous[1])/height, self.scene, self.camera)

    def _scroll(self, window, x, y):
        mujoco.mjv_moveCamera(self.env.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -.05*y, self.scene, self.camera)

    def _update(self, camera):
        mujoco.mjv_updateScene(self.env.model, self.env.data, self.option, None, camera,
                              mujoco.mjtCatBit.mjCAT_ALL, self.scene)

    def capture(self):
        start = time.perf_counter()
        glfw.make_context_current(self.window)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        images, poses = {}, {}
        for name in ('table_camera', 'wrist_camera'):
            cfg = self.cfg[name]
            camera_id = self.env.model.camera(name).id
            self.sensor_camera.fixedcamid = camera_id
            self._update(self.sensor_camera)
            rect = mujoco.MjrRect(0, 0, cfg['width'], cfg['height'])
            mujoco.mjr_render(rect, self.scene, self.context)
            pixels = np.empty((cfg['height'], cfg['width'], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(pixels, None, rect, self.context)
            images[name] = pixels[::-1].copy()
            transform = np.eye(4)
            transform[:3, :3] = self.env.data.cam_xmat[camera_id].reshape(3, 3)
            transform[:3, 3] = self.env.data.cam_xpos[camera_id]
            poses[name] = transform
        self.images = images
        self.env.latest_images = images
        self.env.latest_image_timestamp = self.env.timestamp
        self.capture_ms = (time.perf_counter()-start)*1000
        return images, poses

    def _text(self, rect, left, right='', position=mujoco.mjtGridPos.mjGRID_TOPLEFT):
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, position, rect, left, right, self.context)

    def draw(self, status, images=None):
        glfw.make_context_current(self.window)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.context)
        width, height = glfw.get_framebuffer_size(self.window)
        if width < 40 or height < 40:
            return
        whole = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_rectangle(whole, .065, .08, .105, 1)
        right_width = int(width*.36)
        left_width = width-right_width
        footer = min(285, height//3)
        viewport = mujoco.MjrRect(0, footer, left_width, height-footer)
        self._update(self.camera)
        mujoco.mjr_render(viewport, self.scene, self.context)
        label = 'PANTHERA-HT / ' + status.get('mode', 'JOINT').upper()
        self._text(viewport, label)
        if status.get('recording'):
            indicator = mujoco.MjrRect(15, height-85, min(left_width-30, 550), 38)
            mujoco.mjr_rectangle(indicator, .75, .025, .05, 1)
            self._text(indicator, 'REC  ' + status.get('episode', ''))
        if status.get('paused'):
            self._text(viewport, 'PAUSED - P to resume', position=mujoco.mjtGridPos.mjGRID_BOTTOMLEFT)
        q = np.degrees(self.env.data.qpos[self.env.qids[:6]])
        selected = status.get('selected', 0)
        angles = '   '.join(f'{i+1}{"*" if i == selected else ":"} {v:6.1f}' for i, v in enumerate(q))
        text = '\n'.join([
            f'Joint angles (deg): {angles}',
            f'Sim {status.get("physics_fps", 0):.0f} steps/s | Cameras {status.get("camera_fps", 0):.1f} FPS | Sim/wall {status.get("realtime_factor", 0):.2f}x',
            f'Time {self.env.timestamp:.2f}s | Speed x{status.get("speed", 1):.2f} | IK residual {self.env.last_ik_residual:.3f}',
            '1-6 select joint | Q/E or LEFT/RIGHT move | TAB joint/Cartesian',
            'Cartesian: W/S +X/-X | A/D +Y/-Y | R/F +Z/-Z (world axes)',
            'Orientation: U/J roll | I/K pitch | O/L yaw | Z/X open/close',
            'SPACE record | P pause / emergency stop | H home | BACKSPACE reset',
            '[ / ] slower/faster | ESC quit | Mouse: drag orbit, right pan, wheel zoom',
            status.get('message', '')[:110],
        ])
        self._text(mujoco.MjrRect(10, 0, left_width-15, footer), text)
        frames = images or self.images
        for index, name in enumerate(('table_camera', 'wrist_camera')):
            pane_h = height//2
            panel = mujoco.MjrRect(left_width, height-(index+1)*pane_h, right_width, pane_h)
            if name in frames:
                source = frames[name]
                scale = min(right_width/source.shape[1], (pane_h-38)/source.shape[0])
                w, h = max(1, int(source.shape[1]*scale)), max(1, int(source.shape[0]*scale))
                cache = self.resized_cache.get(name)
                if cache is None or cache[0] is not source or cache[1:3] != (w, h):
                    resized = np.asarray(Image.fromarray(source).resize((w, h), Image.Resampling.BILINEAR))
                    cache = (source, w, h, np.ascontiguousarray(resized[::-1]).ravel())
                    self.resized_cache[name] = cache
                image_rect = mujoco.MjrRect(left_width+(right_width-w)//2, panel.bottom+(pane_h-38-h)//2, w, h)
                mujoco.mjr_drawPixels(cache[3], None, image_rect, self.context)
            self._text(panel, name.replace('_', ' ').upper() + (' / RECORDED' if status.get('replay') else ''))
        glfw.swap_buffers(self.window)

    def close(self):
        if self.window:
            glfw.make_context_current(self.window)
            self.context.free()
            glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None
