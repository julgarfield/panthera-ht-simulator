"""Policy-facing contract. Metres, radians, seconds; Cartesian commands in world axes."""
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np


@dataclass
class Action:
    mode: str = 'joint'
    joint_velocity: np.ndarray = field(default_factory=lambda: np.zeros(6))
    cartesian_twist: np.ndarray = field(default_factory=lambda: np.zeros(6))
    gripper_velocity: float = 0.0

    def validate(self):
        if self.mode not in ('joint', 'cartesian', 'home'):
            raise ValueError(f'Unknown action mode {self.mode}')
        self.joint_velocity = np.asarray(self.joint_velocity, dtype=float)
        self.cartesian_twist = np.asarray(self.cartesian_twist, dtype=float)
        if self.joint_velocity.shape != (6,) or self.cartesian_twist.shape != (6,):
            raise ValueError('Joint velocity and Cartesian twist must each contain six values')
        if not np.isfinite(np.r_[self.joint_velocity, self.cartesian_twist, self.gripper_velocity]).all():
            raise ValueError('Non-finite action rejected')


class RobotInterface(Protocol):
    def get_observation(self) -> dict: ...
    def apply_action(self, action: Action) -> np.ndarray: ...
    def step(self, action: Action) -> dict: ...
