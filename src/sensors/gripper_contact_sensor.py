"""
Gripper contact sensor scaffold.

This module keeps the future sensor implementation and its default
configuration together so it can evolve without adding a separate config file.
The current version only provides a reusable config preset and a lightweight
sensor wrapper skeleton; it does not yet instantiate Genesis native sensors.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import torch

from controllers.backend import Backend


DEFAULT_GRIPPER_CONTACT_SENSOR_CONFIG: Dict[str, Any] = {
    "name": "gripper_contact",
    "enabled": True,
    # Supported intent values for later implementation:
    # - "contact": boolean contact only
    # - "contact_force": contact force only
    # - "hybrid": keep both signals
    "mode": "hybrid",
    "target_entity": "target_object",
    "target_link_names": (
        "base_link",
    ),
    "finger_link_names": (
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger1_2_Link",
        "starlink_space_manipulator_merged.SLDASM9_starlink_space_manipulator_Finger2_1_Link",
    ),
    "require_both_fingers": True,
    "contact_region": {
        "enabled": True,
        "reference_frame": "target_link",
        "reference_link_name": "base_link",
        "offset": (0.0, 0.0, 0.0),
        "tolerance": 0.08,
    },
    "contact": {
        "enabled": True,
        "latched": False,
        "hold_steps": 12,
    },
    "contact_force": {
        "enabled": True,
        "hold_steps": 12,
        "magnitude_threshold": 1e-3,
        "use_local_frame": False,
        "normal_only": False,
        "normal_axis": (0.0, 0.0, 1.0),
        "normal_threshold": None,
        "shear_threshold": None,
    },
    "observation": {
        "include_contact_bool": True,
        "include_force_magnitude": True,
        "include_force_vector": False,
        "include_hold_count": True,
    },
    "debug": {
        "print_on_trigger": False,
        "print_interval": 0,
    },
}


def build_gripper_contact_sensor_config(
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = deepcopy(DEFAULT_GRIPPER_CONTACT_SENSOR_CONFIG)
    if not overrides:
        return config

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


class GripperContactSensor(Backend):
    """
    Future gripper contact sensor wrapper.

    The goal of this wrapper is to expose a stable interface to task code even
    before we finalize which Genesis sensor primitive we use underneath.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = build_gripper_contact_sensor_config(config)
        self.device = torch.device("cpu")
        self.datatype = torch.float32
        self.left_finger_link = None
        self.right_finger_link = None
        self.target_entity = None
        self.contact_active = None
        self.contact_force = None
        self.hold_count = None

    def initialize(self, robot):
        super().initialize(robot)
        if hasattr(robot, "device"):
            self.device = robot.device
        if hasattr(robot, "datatype"):
            self.datatype = robot.datatype

        finger_link_names = self.config.get("finger_link_names", ())
        if len(finger_link_names) >= 2 and hasattr(robot, "robot"):
            self.left_finger_link = robot.robot.get_link(finger_link_names[0])
            self.right_finger_link = robot.robot.get_link(finger_link_names[1])

        self.contact_active = torch.zeros(1, dtype=torch.bool, device=self.device)
        self.contact_force = torch.zeros(1, dtype=self.datatype, device=self.device)
        self.hold_count = torch.zeros(1, dtype=torch.int32, device=self.device)

    def read(self) -> Dict[str, torch.Tensor]:
        return {
            "contact_active": self.contact_active,
            "contact_force": self.contact_force,
            "hold_count": self.hold_count,
        }

    def step(self, dt: float = None):
        # Placeholder: actual Genesis contact/contact-force sensor hookup will
        # be added later.
        return self.read()

    def stop(self):
        pass

    def reset(self):
        if self.contact_active is not None:
            self.contact_active.zero_()
        if self.contact_force is not None:
            self.contact_force.zero_()
        if self.hold_count is not None:
            self.hold_count.zero_()


__all__ = [
    "DEFAULT_GRIPPER_CONTACT_SENSOR_CONFIG",
    "GripperContactSensor",
    "build_gripper_contact_sensor_config",
]
