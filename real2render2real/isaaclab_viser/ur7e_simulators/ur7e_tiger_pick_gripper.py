"""Tiger pick simulator with visual gripper open/close.

Key difference from ur7e_tiger_pick.py:
- Gripper open/close is done by rotating finger mesh prims directly (no physics)
- This gives correct visual appearance for training data without needing
  the gripper to be part of the articulation (which PhysX rejects due to
  nested rigid body hierarchy constraints)
- IK targets 6 arm joints; gripper is animated visually
"""

from dataclasses import dataclass
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

NUM_ARM_JOINTS = 6
FINGER_JOINT_OPEN = 0.0
FINGER_JOINT_CLOSED = 0.7  # visual rotation in radians for finger prims


@dataclass
class PickConfig:
    setup_phase_steps: int = 10
    approach_steps: int = 25
    descend_steps: int = 15
    grasp_steps: int = 7
    lift_steps: int = 25
    hover_steps: int = 10

    approach_height: float = 0.15
    grasp_height: float = 0.06
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
    gripper_closed: bool = False
    tiger_attached: bool = False
    tiger_offset: torch.Tensor = None


class TigerPickGripper(IsaacLabViser):
    def __init__(self, simulation_app, scene_config, **kwargs):
        kwargs.setdefault('init_viser', False)
        self.pick_config = PickConfig()
        self.pick_state = PickState()
        self._sim_cfg_override = None

        super().__init__(simulation_app, scene_config, **kwargs)

        self._apply_pillar_orientation()
        self._spawn_grippers()

        self.isaac_viewport_camera = self.scene.sensors["viewport_camera"]
        self.camera_buffers = {"cam_0": deque(maxlen=1), "cam_1": deque(maxlen=1)}

        self.tiger = self.scene.rigid_objects["tiger"]
        self.tiger_init_pos = torch.tensor(
            (-0.40, 0.05, 0.085), device=self.scene.env_origins.device
        )
        self.tiger_init_rot = torch.tensor(
            (1.0, 0.0, 0.0, 0.0), device=self.scene.env_origins.device
        )

        self.T_base_cam_fixed = torch.tensor(
            [1.375, 1.198, 0.714], device=self.scene.env_origins.device
        )
        self.T_ee_cam_wrist = torch.tensor(
            [0.0, 0.0, 0.05], device=self.scene.env_origins.device
        )

        self.run_simulator()

    def _apply_pillar_orientation(self):
        from pxr import Gf, UsdGeom
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        mat = Gf.Matrix4d(
             0.5054585783556504, -0.01564235936930266, -0.862709071564713,   0,
             0.8614450409941848, -0.0479028782148914,   0.5055865461078429,  0,
            -0.04923481403654982, -0.9987295083515333, -0.010737887813508283, 0,
            -576.3422577523722,    710.4825268723347,    523.6951992568527,   1,
        )
        for env_idx in range(self.scene.num_envs):
            inner_path = f"/World/envs/env_{env_idx}/Pillar/tn__V25_V5xgg2sec0sYY0isSaiJ"
            inner_prim = stage.GetPrimAtPath(inner_path)
            if inner_prim.IsValid():
                xf = UsdGeom.Xformable(inner_prim)
                xf.ClearXformOpOrder()
                op = xf.AddTransformOp()
                op.Set(mat)

    def run_simulator(self):
        self.robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=[".*"],
            body_names=["wrist_3_link"],
        )
        self.robot_entity_cfg.resolve(self.scene)

        self.robot = list(self.scene.articulations.values())[0]
        sim_dt = self.sim.get_physics_dt()
        self.all_joint_names = list(self.robot.data.joint_names)
        self.num_joints = len(self.all_joint_names)
        print(f"[INFO] Robot joints ({self.num_joints}): {self.all_joint_names}")

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
            print("[WARNING] No URDF found, falling back to Jacobian IK")

        self.robot.update(sim_dt)
        self._update_ee_poses()

        count = 0
        self.success_envs = None

        while self.simulation_app.is_running() and self.successful_envs.value < 100:
            sim_start_time = time.time()

            self.robot.update(sim_dt)
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

            if count % 60 == 0:
                clients = self.viser_server.get_clients()
                if clients:
                    self.client = list(clients.values())[0]
                else:
                    self.client = None

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

    def _get_tiger_pose_world(self):
        return self.tiger.data.root_state_w[:, :3]

    # ---- Gripper control ----

    def _spawn_grippers(self):
        """Spawn Robotiq 2F-85 as static visual mesh under wrist_3_link."""
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics
        import os as _os

        _dir = _os.path.dirname(_os.path.realpath(__file__))
        data_dir = _os.path.join(_dir, "../../../data")
        gripper_path = _os.path.abspath(_os.path.join(
            data_dir, "assets/robotiq_2f_85/2F-85/Robotiq_2F_85_edit.usd"
        ))
        pos = (-0.00548, -0.00440, -0.02588)
        rot = (0.70710677, 0.0, 0.0, -0.70710677)

        stage = omni.usd.get_context().get_stage()
        self._gripper_finger_prims = []

        for env_idx in range(self.scene.num_envs):
            parent_path = f"/World/envs/env_{env_idx}/Robot1/wrist_3_link"
            gripper_prim_path = parent_path + "/gripper"

            if not stage.GetPrimAtPath(gripper_prim_path).IsValid():
                prim = UsdGeom.Xform.Define(stage, gripper_prim_path).GetPrim()
                prim.GetReferences().AddReference(gripper_path)

                xf = UsdGeom.Xformable(prim)
                xf.ClearXformOpOrder()
                t_op = xf.AddTranslateOp()
                t_op.Set(Gf.Vec3f(*pos))
                r_op = xf.AddOrientOp()
                r_op.Set(Gf.Quatf(rot[0], Gf.Vec3f(rot[1], rot[2], rot[3])))

                def _walk(pr):
                    for c in pr.GetChildren():
                        try:
                            if c.HasAPI(UsdPhysics.ArticulationRootAPI):
                                c.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                        except Exception:
                            pass
                        try:
                            if c.HasAPI(UsdPhysics.RigidBodyAPI):
                                c.RemoveAPI(UsdPhysics.RigidBodyAPI)
                        except Exception:
                            pass
                        _walk(c)
                _walk(prim)

            # Collect finger prim paths for visual animation
            left_knuckle = stage.GetPrimAtPath(
                gripper_prim_path + "/Robotiq_2F_85/left_outer_knuckle"
            )
            right_knuckle = stage.GetPrimAtPath(
                gripper_prim_path + "/Robotiq_2F_85/right_outer_knuckle"
            )
            self._gripper_finger_prims.append((left_knuckle, right_knuckle))

    def _animate_gripper(self, closed: bool):
        """Visually rotate finger knuckle prims to simulate open/close.

        Writes directly to Fabric (USDRT) which is what the renderer reads
        during simulation. Plain pxr USD writes don't propagate to Fabric.
        """
        import math
        import omni.usd
        import usdrt

        angle = FINGER_JOINT_CLOSED if closed else FINGER_JOINT_OPEN
        cos_half = math.cos(angle / 2)
        sin_half = math.sin(angle / 2)

        if not hasattr(self, '_fabric_stage'):
            self._fabric_stage = usdrt.Usd.Stage.Attach(
                omni.usd.get_context().get_stage_id()
            )

        for left_knuckle, right_knuckle in self._gripper_finger_prims:
            if left_knuckle and left_knuckle.IsValid():
                path = left_knuckle.GetPath().pathString
                rt_prim = self._fabric_stage.GetPrimAtPath(path)
                if rt_prim:
                    attr = rt_prim.GetAttribute("xformOp:orient")
                    if attr:
                        attr.Set(usdrt.Gf.Quatd(cos_half, 0, 0, sin_half))

            if right_knuckle and right_knuckle.IsValid():
                path = right_knuckle.GetPath().pathString
                rt_prim = self._fabric_stage.GetPrimAtPath(path)
                if rt_prim:
                    attr = rt_prim.GetAttribute("xformOp:orient")
                    if attr:
                        attr.Set(usdrt.Gf.Quatd(cos_half, 0, 0, -sin_half))

    def _set_gripper_target(self, joint_pos_des: torch.Tensor, closed: bool):
        """Animate gripper visually (no physics joint control needed)."""
        if not hasattr(self, '_gripper_debug_done'):
            self._gripper_debug_done = True
            left, right = self._gripper_finger_prims[0]
            from pxr import UsdGeom
            if left and left.IsValid():
                xf = UsdGeom.Xformable(left)
                ops = xf.GetOrderedXformOps()
                print(f"[GRIPPER DEBUG] left_knuckle path: {left.GetPath()}")
                print(f"[GRIPPER DEBUG] xform ops: {[(op.GetOpName(), op.GetOpType()) for op in ops]}")
            else:
                print(f"[GRIPPER DEBUG] left_knuckle INVALID")
        self._animate_gripper(closed)

    # ---- Camera ----

    def _set_data_camera_poses(self):
        dev = self.scene.env_origins.device
        fixed_eye = self.T_base_cam_fixed
        fixed_target = torch.tensor([-0.4, 0.25, 0.4], device=dev)

        ee_pos = self.ee_pose_w[0, :3]
        wrist_eye = ee_pos + self.T_ee_cam_wrist
        wrist_target = ee_pos + torch.tensor([0.0, 0.0, -0.2], device=dev)

        eyes = torch.stack([fixed_eye + self.scene.env_origins[0],
                           wrist_eye + self.scene.env_origins[0]], dim=0)
        targets = torch.stack([fixed_target + self.scene.env_origins[0],
                              wrist_target + self.scene.env_origins[0]], dim=0)
        self.isaac_viewport_camera.set_world_poses_from_view(eyes, targets)

    def _render_and_capture(self):
        cam_output = self.isaac_viewport_camera.data.output
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

        if not hasattr(self, '_debugged_poses'):
            self._debugged_poses = True
            self._set_data_camera_poses()
        else:
            self._set_data_camera_poses()

        self.sim.step(render=True)
        self.isaac_viewport_camera.update(0, force_recompute=True)
        self._update_viser()

    def _update_viser(self):
        if not self.init_viser or self.client is None:
            return
        for name in self.urdf_vis.keys():
            robot = self.scene.articulations[name]
            joint_dict = {
                robot.data.joint_names[i]:
                robot.data.joint_pos[self.env][i].item()
                for i in range(len(robot.data.joint_pos[0]))
            }
            self.urdf_vis[name].update_cfg(joint_dict)

        root_pos = self.robot.data.root_state_w[self.env, :3].cpu().numpy()
        root_quat = self.robot.data.root_state_w[self.env, 3:7].cpu().numpy()
        self.base_frame.position = root_pos
        self.base_frame.wxyz = root_quat

        for cam_idx in range(min(2, self.isaac_viewport_camera.cfg.cams_per_env)):
            buf_key = f"cam_{cam_idx}"
            if buf_key in self.camera_buffers and len(self.camera_buffers[buf_key]) > 0:
                cam_data = self.camera_buffers[buf_key][0]
                if "rgb" in cam_data:
                    img = cam_data["rgb"][self.env].cpu().numpy()
                    if cam_idx == 0 and hasattr(self, 'isaac_viewport_viser_handle'):
                        self.isaac_viewport_viser_handle.image = img
                    elif cam_idx == 1 and hasattr(self, 'viser_cam1_handle'):
                        self.viser_cam1_handle.image = img

    # ---- Reset ----

    def _handle_reset(self):
        if self.success_envs is not None:
            print(f"[INFO]: Success Envs: {self.success_envs}")
            if hasattr(self, 'data_logger') and self.data_logger is not None:
                self.data_logger.redir_data(self.success_envs)

        self._reset_robot_state()
        self._reset_tiger_state()
        self.scene.reset()

        self.pick_state.gripper_closed = False
        self.pick_state.tiger_attached = False
        self.pick_state.tiger_offset = None

        if self.use_jaxmp:
            self.controller.reset()

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

    def _reset_tiger_state(self):
        dev = self.scene.env_origins.device
        base_pos = self.tiger_init_pos.expand(self.scene.num_envs, -1).clone()
        base_rot = self.tiger_init_rot.expand(self.scene.num_envs, -1).clone()

        random_xy = (torch.rand((self.scene.num_envs, 2), device=dev) * 2 - 1) * 0.03
        random_z_rot = torch.rand((self.scene.num_envs,), device=dev) * 2 * np.pi

        base_pos[:, 0] += random_xy[:, 0]
        base_pos[:, 1] += random_xy[:, 1]
        base_rot[:, 0] = torch.cos(random_z_rot / 2)
        base_rot[:, 3] = torch.sin(random_z_rot / 2)

        root_state = self.tiger.data.default_root_state.clone()
        root_state[:, :3] = base_pos + self.scene.env_origins
        root_state[:, 3:7] = base_rot
        self.tiger.write_root_state_to_sim(root_state)

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
            self._set_gripper_target(joint_pos_target, closed=False)
            self.robot.set_joint_position_target(joint_pos_target)
            self.robot.write_data_to_sim()

    # ---- Manipulation ----

    def _handle_manipulation(self, count: int):
        cfg = self.pick_config
        offset = count - cfg.setup_phase_steps

        tiger_pos_world = self._get_tiger_pose_world()
        target_world = tiger_pos_world.clone()
        gripper_closed = False

        if offset < cfg.approach_steps:
            t = self._smoothstep(offset / cfg.approach_steps)
            target_world[:, 2] += cfg.approach_height * (1 - t) + cfg.grasp_height * t
        elif offset < cfg.approach_steps + cfg.descend_steps:
            target_world[:, 2] += cfg.grasp_height
        elif offset < cfg.approach_steps + cfg.descend_steps + cfg.grasp_steps:
            target_world[:, 2] += cfg.grasp_height
            gripper_closed = True
            grasp_step = offset - cfg.approach_steps - cfg.descend_steps
            if grasp_step == 0:
                self._attach_tiger()
        elif offset < cfg.approach_steps + cfg.descend_steps + cfg.grasp_steps + cfg.lift_steps:
            t = self._smoothstep(
                (offset - cfg.approach_steps - cfg.descend_steps - cfg.grasp_steps) / cfg.lift_steps
            )
            target_world[:, 2] += cfg.grasp_height * (1 - t) + cfg.lift_height * t
            gripper_closed = True
        else:
            target_world[:, 2] += cfg.lift_height
            gripper_closed = True

        self.pick_state.gripper_closed = gripper_closed

        if self.pick_state.tiger_attached and self.pick_state.tiger_offset is not None:
            self._move_tiger_with_ee()

        # IK: world → robot base frame
        root_pos_w = self.robot.data.root_state_w[:, :3]
        root_quat_w = self.robot.data.root_state_w[:, 3:7]
        target_identity_quat = torch.zeros_like(root_quat_w)
        target_identity_quat[:, 0] = 1.0
        target_pos_b, _ = subtract_frame_transforms(
            root_pos_w, root_quat_w,
            target_world, target_identity_quat,
        )

        if self.use_jaxmp:
            joint_pos_des = self._solve_ik_jaxmp(target_pos_b, count)
        else:
            joint_pos_des = self._solve_ik_jacobian(target_pos_b)

        # Set gripper target on the full joint vector
        self._set_gripper_target(joint_pos_des, closed=gripper_closed)

        self.robot.set_joint_position_target(joint_pos_des)
        self.robot.write_data_to_sim()

    # ---- Tiger attachment ----

    def _attach_tiger(self):
        ee_pos = self.ee_pose_w[:, :3]
        tiger_pos = self._get_tiger_pose_world()

        dist = torch.norm(ee_pos - tiger_pos, dim=-1)
        if dist[0] > 0.20:
            print(f"[WARN] Skip tiger attach: EE-tiger dist = {dist[0]:.4f}m (> 0.20m)")
            return

        ee_quat = self.ee_pose_w[:, 3:7]
        tiger_quat = self.tiger.data.root_state_w[:, 3:7]

        ee_tf = tf.SE3(torch.cat([ee_quat, ee_pos], dim=-1))
        tiger_tf = tf.SE3(torch.cat([tiger_quat, tiger_pos], dim=-1))

        offset_tf = ee_tf.inverse() @ tiger_tf
        self.pick_state.tiger_offset = offset_tf.wxyz_xyz
        self.pick_state.tiger_attached = True
        print(f"[INFO] Tiger attached! dist = {dist[0]:.4f}m")

    def _move_tiger_with_ee(self):
        ee_pos = self.ee_pose_w[:, :3]
        ee_quat = self.ee_pose_w[:, 3:7]

        ee_tf = tf.SE3(torch.cat([ee_quat, ee_pos], dim=-1))
        offset_tf = tf.SE3(self.pick_state.tiger_offset)

        new_tiger_tf = ee_tf @ offset_tf
        new_pos_w = new_tiger_tf.wxyz_xyz[:, 4:]
        new_rot = new_tiger_tf.wxyz_xyz[:, :4]

        root_state = self.tiger.data.root_state_w.clone()
        root_state[:, :3] = new_pos_w
        root_state[:, 3:7] = new_rot
        self.tiger.write_root_state_to_sim(root_state)

    # ---- IK solvers ----

    def _solve_ik_jaxmp(self, target_pos_b, count):
        _, ee_quat_b = self._get_ee_poses()

        target_poses = np.zeros((self.scene.num_envs, 1, 7))
        target_poses[:, 0, :3] = target_pos_b.cpu().numpy()
        target_poses[:, 0, 3:] = ee_quat_b.cpu().numpy()

        joints_jmp = self.controller.compute_ik(target_poses)
        joints = np.array(joints_jmp)

        joint_pos_des = torch.zeros(
            (self.scene.num_envs, self.num_joints),
            device=self.robot.device,
        )
        joint_pos_des[:, :NUM_ARM_JOINTS] = torch.tensor(
            joints[:, :NUM_ARM_JOINTS], device=self.robot.device
        )
        return joint_pos_des.clamp_(
            self.robot.data.soft_joint_pos_limits[..., 0],
            self.robot.data.soft_joint_pos_limits[..., 1],
        )

    def _solve_ik_jacobian(self, target_pos_b):
        joint_pos = self.robot.data.joint_pos.clone()
        joint_pos_des = joint_pos.clone()

        step_size = 0.3
        for _ in range(3):
            self.robot.update(sim_dt=0.0)
            ee_b, _ = self._get_ee_poses()
            delta = target_pos_b - ee_b
            delta[:, 2] += self.pick_config.grasp_height
            joint_pos_des[:, :NUM_ARM_JOINTS] += (
                step_size * delta * 0.1
            ).expand(-1, NUM_ARM_JOINTS)

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

    # ---- Viser overrides ----

    def _setup_viser_gui(self):
        super()._setup_viser_gui()
        with self.viser_server.gui.add_folder("Wrist Camera (cam_1)"):
            self.viser_cam1_handle = self.viser_server.gui.add_image(
                np.zeros((240, 320, 3))
            )

    # ---- Utilities ----

    @staticmethod
    def _smoothstep(t):
        t = max(0.0, min(1.0, t))
        return t * t * (3 - 2 * t)
