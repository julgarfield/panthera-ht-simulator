from collections import deque
import glfw
import numpy as np
from .interfaces import Action


class KeyboardController:
    def __init__(self, window, cfg):
        self.cfg = cfg
        self.held = set()
        self.events = deque()
        self.mode = 'joint'
        self.selected = 0
        self.speed = 1.0
        self.homing = False
        glfw.set_key_callback(window, self.on_key)
        glfw.set_window_focus_callback(window, self.on_focus)

    def on_focus(self, window, focused):
        self.held.clear()
        if not focused:
            self.events.append('focus_lost')

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.RELEASE:
            self.held.discard(key)
            return
        if action != glfw.PRESS:
            return
        self.held.add(key)
        if glfw.KEY_1 <= key <= glfw.KEY_6:
            self.selected = key-glfw.KEY_1
        elif key == glfw.KEY_TAB:
            self.mode = 'cartesian' if self.mode == 'joint' else 'joint'
            self.homing = False
            self.events.append('hold')
        elif key == glfw.KEY_LEFT_BRACKET:
            self.speed = max(.125, self.speed/2)
        elif key == glfw.KEY_RIGHT_BRACKET:
            self.speed = min(2.0, self.speed*2)
        else:
            event = {glfw.KEY_SPACE: 'record', glfw.KEY_P: 'pause', glfw.KEY_ESCAPE: 'quit',
                     glfw.KEY_BACKSPACE: 'reset', glfw.KEY_H: 'home'}.get(key)
            if event:
                self.events.append(event)

    def _axis(self, positive, negative):
        return int(positive in self.held)-int(negative in self.held)

    def get_action(self, observation):
        c = self.cfg['controls']
        grip = self._axis(glfw.KEY_Z, glfw.KEY_X)*c['gripper_speed']*self.speed
        if self.homing:
            if np.max(np.abs(observation['joint_positions']-np.asarray(self.cfg['robot']['home_configuration']))) < .03:
                self.homing = False
            else:
                return Action(mode='home', gripper_velocity=grip)
        if self.mode == 'joint':
            velocity = np.zeros(6)
            direction = np.clip(self._axis(glfw.KEY_E, glfw.KEY_Q)+self._axis(glfw.KEY_RIGHT, glfw.KEY_LEFT), -1, 1)
            velocity[self.selected] = direction*c['joint_speed']*self.speed
            return Action(joint_velocity=velocity, gripper_velocity=grip)
        twist = np.array([self._axis(glfw.KEY_W, glfw.KEY_S), self._axis(glfw.KEY_A, glfw.KEY_D),
                          self._axis(glfw.KEY_R, glfw.KEY_F), self._axis(glfw.KEY_U, glfw.KEY_J),
                          self._axis(glfw.KEY_I, glfw.KEY_K), self._axis(glfw.KEY_O, glfw.KEY_L)], dtype=float)
        twist[:3] *= c['cartesian_translation_speed']*self.speed
        twist[3:] *= c['cartesian_rotation_speed']*self.speed
        return Action(mode='cartesian', cartesian_twist=twist, gripper_velocity=grip)
