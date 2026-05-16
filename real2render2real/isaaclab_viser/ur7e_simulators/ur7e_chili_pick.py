"""Scripted chili pick simulator for UR5e (proxy for UR7e).

Follows the Franka CoffeeMaker pattern (single-arm + gripper).
Uses JaxMP differential IK for joint control.
Scripted trajectory: approach -> descend -> grasp -> lift.
Kinematic attachment for chili during grasp phase.

Headless mode: no viser, captures camera data directly.
"""

from dataclasses import dataclass, field
from typing import Dict
import torch
import numpy as np
import time
from collections import deque
from pathlib import Path

from real2render2real.isaaclab_viser.base import IsaacLabViser
from real2render2real.isaaclab_viser.controllers.jaxmp_diff_ik_controller import (
    JaxMPBatchedController,
)
import real2render2real.utils.transforms as tf
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaacsim.core.prims import XFormPrim

NUM_ARM_JOINTS = 6


@dataclass
class PickConfig:
    """Phase timing for scripted pick trajectory."""
    setup_phase_steps: int = 10
    approach_steps: int = 25
    descend_steps: int = 15
    grasp_steps: int = 7
    lift_steps: int = 25
    hover_steps: int = 10

    # EE height offsets relative to object (negative = above)
    approach_height: float = 0.15
    grasp_height: float = 0.03
    lift_height: float = 0.25

    @property
    def grasp_start(self) -> int:
        return self.setup_phase_steps + self.approach_steps + self.descend_steps

    @property
    def grasp_end(self) -> int:
        return self.grasp_start + self.grasp_steps

    @property
    def total_steps(self) -> int:
        return (self.setup_phase_steps + self.approach_steps +
                self.descend_steps + self.grasp_steps +
                self.lift_steps + self.hover_steps)


@dataclass
class PickState:
    """Runtime state for the pick trajectory."""
    gripper_closed: bool = False
    chili_attached: bool = False
    chili_offset: torch.Tensor = None  # SE3 offset from EE to chili


class ChiliPick(IsaacLabViser):
    def __init__(self, simulation_app, scene_config, **kwargs):
        kwargs.setdefault('init_viser', False)
        self.pick_config = PickConfig()
        self.pick_state = PickState()

        # Lightweight render config to reduce VRAM
        self._sim_cfg_override = None

        super().__init__(simulation_app, scene_config, **kwargs)

        self.isaac_viewport_camera = self.scene.sensors["viewport_camera"]
        self.camera_buffers = {"cam_0": deque(maxlen=1), "cam_1": deque(maxlen=1)}

        # Per-env XFormPrim for chili (kinematic, no physics needed)
        self.chili_prims = [
            XFormPrim(f"/World/envs/env_{i}/chili") for i in range(self.scene.num_envs)
        ]
        self.chili_state = torch.zeros(
            (self.scene.num_envs, 7), device=self.scene.env_origins.device
        )

        # Camera extrinsics from hand-eye calibration
        # Fixed D435I relative to robot base (world frame, already in meters)
        self.T_base_cam_fixed = torch.tensor(
            [0.122, -0.898, 0.451], device=self.scene.env_origins.device
        )
        # Wrist D405 relative to end-effector (meters)
        self.T_ee_cam_wrist = torch.tensor(
            [-0.0057, -0.0619, 0.0401], device=self.scene.env_origins.device
        )

        self.run_simulator()

    def run_simulator(self):
        """Main simulation loop."""
        # --- IK Controller ---
        self.robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=[".*"],
            body_names=["wrist_3_link"],
        )
        self.robot_entity_cfg.resolve(self.scene)

        self.robot = list(self.scene.articulations.values())[0]
        self.all_joint_names = list(self.robot.data.joint_names)
        self.num_joints = len(self.all_joint_names)
        print(f"[INFO] Robot joints ({self.num_joints}): {self.all_joint_names}")

        # JaxMP IK controller (6 DOF arm only)
        urdf_path = self.urdf_path.get('robot') if self.urdf_path else None
        if urdf_path and Path(urdf_path).exists():
            self.controller = JaxMPBatchedController(
                urdf_path=urdf_path,
                num_envs=self.scene.num_envs,
                num_ees=1,
                target_names=["tool0_joint"],
                home_pose=self.robot.data.default_joint_pos[0, :NUM_ARM_JOINTS].cpu().numpy(),
            )
            self.use_jaxmp = True
            print("[INFO] JaxMP IK controller initialized")
        else:
            self.use_jaxmp = False
            print("[WARNING] No URDF found for JaxMP IK, falling back to Jacobian method")

        # --- Main loop ---
        count = 0
        sim_dt = self.sim.get_physics_dt()
        self.success_envs = None

        while self.simulation_app.is_running() and self.successful_envs.value < 100:
            sim_start_time = time.time()

            self._update_ee_poses()
            self._render_and_capture()

            if count % self.pick_config.total_steps == 0:
                self._handle_reset()
                count = 0

            if count > self.pick_config.setup_phase_steps:
                self._handle_manipulation(count)
                self._log_data(count)
            else:
                self._handle_setup_phase(count)

            count += 1
            self.sim_step_time_ms.value = (time.time() - sim_start_time) * 1e3
            # Non-blocking viser client check (once per second)
            if count % 60 == 0 and len(self.viser_server.get_clients()) > 0:
                self.client = self.viser_server.get_clients()[0]

    # ---- EE tracking ----

    def _update_ee_poses(self):
        self.ee_pose_w = self.robot.data.body_state_w[
            :, self.robot_entity_cfg.body_ids[0], 0:7
        ]

    def _get_ee_poses(self):
        ee_pose = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose = self.robot.data.root_state_w[:, 0:7]
        return subtract_frame_transforms(
            root_pose[:, 0:3], root_pose[:, 3:7],
            ee_pose[:, 0:3], ee_pose[:, 3:7],
        )

    # ---- Camera ----

    def _set_data_camera_poses(self):
        """Set cam_0 (fixed D435I) and cam_1 (wrist D405) using eye/target pairs."""
        dev = self.scene.env_origins.device

        # Cam 0: Fixed D435I — higher overview to see pillar + robot
        robot_base = self.robot.data.root_state_w[0, :3]
        fixed_eye = torch.tensor([1.5, 0.3, 2.5], device=dev)   # right side, higher up
        fixed_target = robot_base                                # look at robot base

        # Cam 1: Wrist D405 looking slightly ahead from EE
        ee_pos = self.ee_pose_w[0, :3]
        ee_forward = torch.tensor([0.0, 0.0, -0.3], device=dev)  # look 30cm forward (-Z in EE frame ≈ downward)
        wrist_eye = ee_pos + self.T_ee_cam_wrist
        wrist_target = ee_pos + ee_forward

        # Stack for all camera instances: [cam0, cam1]
        eyes = torch.stack([fixed_eye + self.scene.env_origins[0],
                           wrist_eye + self.scene.env_origins[0]], dim=0)
        targets = torch.stack([fixed_target + self.scene.env_origins[0],
                              wrist_target + self.scene.env_origins[0]], dim=0)

        self.isaac_viewport_camera.set_world_poses_from_view(eyes, targets)

    def _render_and_capture(self):
        self._set_data_camera_poses()
        self.sim.render()
        cam_output = self.isaac_viewport_camera.data.output
        # Debug: check cam poses once
        if not hasattr(self, '_debugged_poses'):
            self._debugged_poses = True
            cam0_pos = self.isaac_viewport_camera._view.get_world_poses()
            print(f"[DEBUG] cam_0 world pos: {cam0_pos[0][0].tolist()}")
            print(f"[DEBUG] cam_1 world pos: {cam0_pos[0][1].tolist()}")
        cams_per_env = self.isaac_viewport_camera.cfg.cams_per_env
        num_envs = self.scene.num_envs

        for cam_idx in range(cams_per_env):
            indices = list(range(cam_idx, num_envs * cams_per_env, cams_per_env))
            cam_data = {}
            for key in cam_output.keys():
                cam_data[key] = cam_output[key][indices].clone()
            buf_key = f"cam_{cam_idx}"
            if buf_key not in self.camera_buffers:
                self.camera_buffers[buf_key] = deque(maxlen=1)
            self.camera_buffers[buf_key].append(cam_data)

    # ---- Reset ----

    def _handle_reset(self):
        if self.success_envs is not None:
            print(f"[INFO]: Success Envs: {self.success_envs}")
            if hasattr(self, 'data_logger') and self.data_logger is not None:
                self.data_logger.redir_data(self.success_envs)

        self._reset_robot_state()
        self._reset_object_state()
        self.scene.reset()

        # Reset pick state
        self.pick_state.gripper_closed = False
        self.pick_state.chili_attached = False
        self.pick_state.chili_offset = None

        self.randomize_lighting()
        self.randomize_viewaug()

        print("[INFO]: Resetting state...")
        self.success_envs = torch.ones(
            (self.scene.num_envs,), device=self.scene.env_origins.device, dtype=bool
        )

    def _reset_robot_state(self):
        root_state = self.robot.data.default_root_state.clone()
        root_state[:, :3] += self.scene.env_origins
        self.robot.write_root_state_to_sim(root_state)
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

    def _reset_object_state(self):
        # Default chili position on the table (from scene config: pos=(0.4, 0.0, TABLE_HEIGHT+0.05))
        default_pos = torch.tensor(
            (0.4, 0.0, 0.84), device=self.scene.env_origins.device
        )
        default_rot = torch.tensor(
            (1.0, 0.0, 0.0, 0.0), device=self.scene.env_origins.device
        )

        base_pos = default_pos.expand(self.scene.num_envs, -1).clone()
        base_rot = default_rot.expand(self.scene.num_envs, -1).clone()

        # Randomize chili XY position and Z rotation
        random_xy = (torch.rand((self.scene.num_envs, 2), device=base_pos.device) * 2 - 1) * 0.06
        random_z_rot = torch.rand((self.scene.num_envs,), device=base_pos.device) * 2 * np.pi

        base_pos[:, 0] += random_xy[:, 0]
        base_pos[:, 1] += random_xy[:, 1]
        base_rot[:, 0] = torch.cos(random_z_rot / 2)
        base_rot[:, 3] = torch.sin(random_z_rot / 2)

        # Store local state
        self.chili_state[:, :3] = base_pos
        self.chili_state[:, 3:7] = base_rot

        # Write world poses via XFormPrim (per env, needs batch dim torch tensor)
        pos_world = base_pos + self.scene.env_origins
        for i in range(self.scene.num_envs):
            self.chili_prims[i].set_world_poses(
                positions=pos_world[i].clone().detach().reshape(1, 3),
                orientations=base_rot[i].clone().detach().reshape(1, 4),
            )

    # ---- Setup phase ----

    def _handle_setup_phase(self, count: int):
        if count < 3:
            joint_pos_target = (
                self.robot.data.default_joint_pos
                + torch.randn_like(self.robot.data.joint_pos) * 0.01
            ).clamp_(
                self.robot.data.soft_joint_pos_limits[..., 0],
                self.robot.data.soft_joint_pos_limits[..., 1],
            )
            self.robot.set_joint_position_target(joint_pos_target)
            self.robot.write_data_to_sim()

        self.sim.step(render=False)

    # ---- Manipulation ----

    def _handle_manipulation(self, count: int):
        cfg = self.pick_config
        offset = count - cfg.setup_phase_steps

        # Get chili position (local coords)
        obj_pos = self.chili_state[:, :3]

        # Determine target EE position
        target_pos = obj_pos.clone()
        gripper_closed = False

        if offset < cfg.approach_steps:
            # Phase 1: Approach from above
            t = self._smoothstep(offset / cfg.approach_steps)
            target_pos[:, 2] += cfg.approach_height * (1 - t) + cfg.grasp_height * t
        elif offset < cfg.approach_steps + cfg.descend_steps:
            # Phase 2: Descend to grasp
            t = self._smoothstep((offset - cfg.approach_steps) / cfg.descend_steps)
            target_pos[:, 2] += cfg.grasp_height
        elif offset < cfg.approach_steps + cfg.descend_steps + cfg.grasp_steps:
            # Phase 3: Grasp (close gripper)
            target_pos[:, 2] += cfg.grasp_height
            gripper_closed = True
            grasp_step = offset - cfg.approach_steps - cfg.descend_steps
            if grasp_step == 0:
                # Attach chili to EE kinematically
                self._attach_chili()
        elif offset < cfg.approach_steps + cfg.descend_steps + cfg.grasp_steps + cfg.lift_steps:
            # Phase 4: Lift
            t = self._smoothstep(
                (offset - cfg.approach_steps - cfg.descend_steps - cfg.grasp_steps) / cfg.lift_steps
            )
            target_pos[:, 2] += cfg.grasp_height * (1 - t) + cfg.lift_height * t
            gripper_closed = True
        else:
            # Phase 5: Hover at top
            target_pos[:, 2] += cfg.lift_height
            gripper_closed = True

        self.pick_state.gripper_closed = gripper_closed

        # Move chili with EE if attached
        if self.pick_state.chili_attached and self.pick_state.chili_offset is not None:
            self._move_chili_with_ee()

        # Compute joint positions via IK
        if self.use_jaxmp:
            joint_pos_des = self._solve_ik_jaxmp(target_pos, count)
        else:
            joint_pos_des = self._solve_ik_jacobian(target_pos)

        self.robot.set_joint_position_target(joint_pos_des)
        self.robot.write_data_to_sim()
        self.sim.step(render=False)

    # ---- Chili attachment ----

    def _attach_chili(self):
        """Compute and store the offset from EE to chili for kinematic attachment."""
        ee_pos = self.ee_pose_w[:, :3]          # world coords
        ee_quat = self.ee_pose_w[:, 3:7]        # world coords
        obj_pos = self.chili_state[:, :3] + self.scene.env_origins   # local -> world
        obj_quat = self.chili_state[:, 3:7]                           # same in world

        # Compute offset in EE frame
        ee_tf = tf.SE3(torch.cat([ee_quat, ee_pos], dim=-1))
        obj_tf = tf.SE3(torch.cat([obj_quat, obj_pos], dim=-1))

        # chili_offset = ee_tf.inv() @ obj_tf
        offset_tf = ee_tf.inverse() @ obj_tf
        self.pick_state.chili_offset = offset_tf.wxyz_xyz  # (N, 7)
        self.pick_state.chili_attached = True

    def _move_chili_with_ee(self):
        """Move chili pose to follow EE using stored offset."""
        ee_pos = self.ee_pose_w[:, :3]
        ee_quat = self.ee_pose_w[:, 3:7]

        ee_tf = tf.SE3(torch.cat([ee_quat, ee_pos], dim=-1))
        offset_tf = tf.SE3(self.pick_state.chili_offset)

        new_chili_tf = ee_tf @ offset_tf
        new_pos_w = new_chili_tf.wxyz_xyz[:, 4:]  # World position
        new_rot = new_chili_tf.wxyz_xyz[:, :4]    # Quaternion (w, x, y, z)

        # Update tracked state (local coords)
        self.chili_state[:, :3] = new_pos_w - self.scene.env_origins
        self.chili_state[:, 3:7] = new_rot

        # Write to XFormPrim (world coords, per env, needs batch dim torch tensor)
        for i in range(self.scene.num_envs):
            self.chili_prims[i].set_world_poses(
                positions=new_pos_w[i].clone().detach().reshape(1, 3),
                orientations=new_rot[i].clone().detach().reshape(1, 4),
            )

    # ---- IK solvers ----

    def _solve_ik_jaxmp(self, target_pos, count):
        """Solve IK using JaxMP differential IK controller."""
        # Build target poses (N, 1, 7) in xyz_wxyz format
        ee_quat_b = self._get_ee_poses()[1]
        # Default EE orientation: gripper pointing down
        down_quat = torch.tensor([1, 0, 0, 0], device=target_pos.device).unsqueeze(0).expand(
            self.scene.num_envs, -1
        )

        target_poses = np.zeros((self.scene.num_envs, 1, 7))
        target_poses[:, 0, :3] = target_pos.cpu().numpy()
        target_poses[:, 0, 3:] = down_quat.cpu().numpy()

        joints_jmp = self.controller.compute_ik(target_poses)
        joints = np.array(joints_jmp)

        joint_pos_des = torch.zeros(
            (self.scene.num_envs, self.num_joints),
            device=self.robot.device,
        )
        joint_pos_des[:, :NUM_ARM_JOINTS] = torch.tensor(joints[:, :NUM_ARM_JOINTS])

        return joint_pos_des.clamp_(
            self.robot.data.soft_joint_pos_limits[..., 0],
            self.robot.data.soft_joint_pos_limits[..., 1],
        )

    def _solve_ik_jacobian(self, target_pos):
        """Fallback IK using iterative Jacobian approach."""
        joint_pos = self.robot.data.joint_pos[:, :self.num_joints].clone()
        joint_pos_des = joint_pos.clone()

        step_size = 0.3
        for _ in range(3):
            self.robot.update(sim_dt=0.0)
            ee_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], :3]
            delta = target_pos - (ee_w - self.scene.env_origins)
            delta[:, 2] += self.pick_config.grasp_height
            joint_pos_des += step_size * delta.unsqueeze(1).expand(-1, self.num_joints) * 0.1

        return joint_pos_des.clamp_(
            self.robot.data.soft_joint_pos_limits[..., 0],
            self.robot.data.soft_joint_pos_limits[..., 1],
        )

    # ---- Data logging ----

    def _log_data(self, count: int):
        if self.data_logger is None:
            return

        ee_pos_b, ee_quat_b = self._get_ee_poses()

        robot_data = {
            "joint_names": list(self.robot.data.joint_names[:NUM_ARM_JOINTS]),
            "joint_angles": self.robot.data.joint_pos[:, :NUM_ARM_JOINTS]
                .clone().cpu().detach().numpy(),
            "ee_pos": torch.cat([ee_pos_b, ee_quat_b], dim=1)
                .cpu().detach().numpy(),
            "gripper_binary_cmd": torch.full(
                (self.scene.num_envs, 1),
                1.0 if self.pick_state.gripper_closed else 0.0,
                device=self.robot.device,
            ).cpu().detach().numpy(),
        }

        self.data_logger.save_data(
            self.camera_buffers,
            robot_data,
            count - self.pick_config.setup_phase_steps - 1,
            self.output_dir,
        )

        stats = self.data_logger.get_stats()
        self.save_time_ms.value = int(stats["save_time"] * 1e3)
        self.images_per_second.value = stats['images_per_second']
        self.successful_envs.value = stats['total_successful_envs']

    # ---- Utilities ----

    @staticmethod
    def _smoothstep(t):
        t = max(0.0, min(1.0, t))
        return t * t * (3 - 2 * t)
