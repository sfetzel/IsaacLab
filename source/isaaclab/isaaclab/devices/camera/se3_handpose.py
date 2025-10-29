# Based on se3_keyboard.py.
# Simon Fetzel <simon.fetzel@stuba.sk>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hand pose controller for SE(3) control."""

import numpy as np
import torch
import weakref
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

import carb
import omni

from ..device_base import DeviceBase, DeviceCfg

#from lacny_teleop import HandPoseEstimator
from lacny_teleop import CircleEstimator, GripperState, HandPoseEstimator

@dataclass
class Se3HandPoseCfg(DeviceCfg):
    """Configuration for SE3 hand pose."""

    gripper_term: bool = True
    pos_sensitivity: float = 0.04
    rot_sensitivity: float = 0.08
    retargeters: None = None
    

class Se3HandPose(DeviceBase):
    """A hand pose controller for sending SE(3) commands as delta poses and binary command (open/close).

    """

    def __init__(self, cfg: Se3HandPoseCfg):
        """Initialize the hand pose layer.

        Args:
            cfg: Configuration object for hand pose settings.
        """
        # store inputs
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.gripper_term = cfg.gripper_term
        self._sim_device = cfg.sim_device
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()

        # command buffers
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        self.counter = 0
        self.decay = 0.95
        # dictionary for additional callbacks
        self._additional_callbacks = dict()
        self._estimator = HandPoseEstimator(0)
        self._estimator.is_paused = True
        self._estimator.start()

    def __del__(self):
        """Release the keyboard interface."""
        self._estimator.stop()

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Hand Pose Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\tTry your luck"
        return msg

    """
    Operations
    """

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func
        
    def reset(self):
        # default flags
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)


    def advance(self) -> torch.Tensor:
        """Provides the result from keyboard event state.

        Returns:
            torch.Tensor: A 7-element tensor containing:
                - delta pose: First 6 elements as [x, y, z, rx, ry, rz] in meters and radians.
                - gripper command: Last element as a binary value (+1.0 for open, -1.0 for close).
        """
        deltas = self._estimator.get_deltas()
        if not deltas is None:
            self._delta_pos += deltas[:3] * self.pos_sensitivity
            self._delta_rot += deltas[3:6] * self.rot_sensitivity
            self._close_gripper = deltas[-1] == GripperState.Closed.value
            
        # convert to rotation vector
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        # return the command and gripper state
        command = np.concatenate([self._delta_pos, rot_vec])
        if self.gripper_term:
            gripper_value = -1.0 if self._close_gripper else 1.0
            command = np.append(command, gripper_value)

        command_tensor = torch.tensor(command, dtype=torch.float32, device=self._sim_device)
        self._delta_pos *= self.decay
        self._delta_rot *= self.decay
        return command_tensor

