# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def pre_grasp(
    env: ManagerBasedRLEnv,
    distance_std: float,
    rotation_std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["left_carriage_joint"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward configurations that favor a grasp-ready pose.

    The reward becomes larger when:
    - the object is inside a small grasp volume around the end-effector, and
    - the gripper is closing.
    """
    robot = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # Gripper openness in [0, 1] using the configured joint limits.
    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    gripper_lower = robot.data.soft_joint_pos_limits[:, robot_cfg.joint_ids, 0]
    gripper_upper = robot.data.soft_joint_pos_limits[:, robot_cfg.joint_ids, 1]
    gripper_opening = torch.clamp((gripper_pos - gripper_lower) / (gripper_upper - gripper_lower).clamp(min=1.0e-6), 0.0, 1.0)
    gripper_opening = gripper_opening.mean(dim=1)

    # Object position relative to the end-effector.
    object_pos_w = object.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    object_pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_quat_w, object_pos_w)
    grasp_distance = torch.norm(object_pos_ee, dim=-1)
    grasp_volume_score = torch.sigmoid((distance_std - grasp_distance) / (distance_std * 0.25))

    # Larger is better: object inside the grasp volume + more closing.
    gripper_closure = torch.clamp(1.0 - gripper_opening, 0.0, 1.0)
    return grasp_volume_score * gripper_closure

def object_inside_gripper_cylinder(
    env: ManagerBasedRLEnv,
    radius: float = 0.03,
    min_h: float = 0.0,
    max_h_when_closed: float = 0.06,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["left_carriage_joint"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    # object pos in EE frame
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_quat_w, object_pos_w)  # (N,3)

    device = pos_ee.device
    # Hardcode axis = Y (green vector / gripper motion direction)
    axis = torch.tensor([0.0, 1.0, 0.0], device=device)

    # height along axis (signed) and radial distance in X-Z plane
    h_along = (pos_ee * axis).sum(dim=1)  # (N,)
    radial_vec = pos_ee - h_along.unsqueeze(-1) * axis
    radial_dist = torch.norm(radial_vec, dim=1)

    # --- compute gripper opening in meters using joint positions and soft limits ---
    robot = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]  # (N, n_joints)
    gripper_lower = robot.data.soft_joint_pos_limits[:, robot_cfg.joint_ids, 0]
    gripper_upper = robot.data.soft_joint_pos_limits[:, robot_cfg.joint_ids, 1]

    span_per_joint = (gripper_upper - gripper_lower).clamp(min=1.0e-6)  # max possible displacement per joint
    # opening displacement measured from closed (lower) position, per-joint
    opening_disp_per_joint = (gripper_pos - gripper_lower).clamp(min=0.0)
    # aggregate across joints (mean)
    opening_m = opening_disp_per_joint.mean(dim=1)  # (N,)
    span_max = span_per_joint.mean(dim=1).clamp(min=1.0e-6)  # (N,)
    opening_frac = (opening_m / span_max).clamp(0.0, 1.0)

    # half-width of the gripper aperture (we treat opening_m as full aperture from closed)
    half_aperture = opening_m * 0.5  # (N,)

    # dynamic height mapping (optional): use normalized opening to adapt cylinder height
    dynamic_h = min_h + (1.0 - opening_frac) * (max_h_when_closed - min_h)  # (N,)

    # --- Y-axis inclusion check: object y must lie within [-half_aperture, +half_aperture] ---
    y_coord = pos_ee[:, 1]  # (N,)
    # continuous y-score (higher when deeper inside aperture); use sigmoid for smoothness
    # avoid div-by-zero by clamping half_aperture
    half_clamped = half_aperture.clamp(min=1.0e-6)
    y_score = torch.sigmoid((half_clamped - torch.abs(y_coord)) / (half_clamped * 0.2))

    # radial proximity score (X-Z plane)
    radial_score = torch.sigmoid((radius - radial_dist) / (radius * 0.2))

    # final continuous score: radial * y * (optionally height/closure influence)
    # here we also require the object to be between 0..dynamic_h along the Y-axis direction
    within_height = (h_along >= 0.0) & (h_along <= dynamic_h)
    final_score = radial_score * y_score * within_height.float()

    return final_score

def object_inside_gripper_binary(
    env: ManagerBasedRLEnv,
    radius: float = 0.01,
    min_h: float = 0.0,
    max_h_when_closed: float = 0.06,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["left_carriage_joint"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Binary check (1.0 inside, 0.0 outside) using Y-axis aperture + circular cross-section + dynamic height."""
    # positions in EE frame
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_pos_w = obj.data.root_pos_w[:, :3]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_quat_w, object_pos_w)  # (N,3)

    device = pos_ee.device
    axis = torch.tensor([0.0, 1.0, 0.0], device=device)  # Y fixed

    # height along axis and radial distance in X-Z plane
    h_along = (pos_ee * axis).sum(dim=1)  # (N,)
    radial_vec = pos_ee - h_along.unsqueeze(-1) * axis
    radial_dist = torch.norm(radial_vec, dim=1)

    # compute gripper opening displacement from joint positions (aggregate)
    robot = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]  # (N, n_joints)
    gripper_lower = robot.data.soft_joint_pos_limits[:, robot_cfg.joint_ids, 0]
    gripper_upper = robot.data.soft_joint_pos_limits[:, robot_cfg.joint_ids, 1]

    span_per_joint = (gripper_upper - gripper_lower).clamp(min=1.0e-6)
    opening_disp_per_joint = (gripper_pos - gripper_lower).clamp(min=0.0)
    opening_m = opening_disp_per_joint.mean(dim=1)  # (N,)
    span_max = span_per_joint.mean(dim=1).clamp(min=1.0e-6)
    opening_frac = (opening_m / span_max).clamp(0.0, 1.0)

    half_aperture = opening_m * 0.5  # (N,)

    # dynamic height (closed -> taller, open -> shorter)
    dynamic_h = min_h + (1.0 - opening_frac) * (max_h_when_closed - min_h)  # (N,)

    # inclusion checks
    y_coord = pos_ee[:, 1]  # (N,)
    inside_y = torch.abs(y_coord) <= half_aperture
    inside_radial = radial_dist <= radius
    within_height = (h_along >= 0.0) & (h_along <= dynamic_h)

    mask = inside_y & inside_radial & within_height
    return mask.float()  # Tensor shape (N,)