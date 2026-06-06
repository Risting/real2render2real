"""Spatial perception cube capture simulator.

Both robot arms are ArticulationCfg.  Supports two modes:
  --arm_poses arm_poses.json  → load 100 pose groups, outer loop over poses
  (no --arm_poses)            → use hardcoded defaults (backward compatible)
"""

import json
import os
import time

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from real2render2real.isaaclab_viser.spatial_perception.camera_extrinsics import (
    get_fixed_cam_poses,
)


class SpatialCubeCapture:
    """Standalone simulator for capturing spatial perception dataset images."""

    VOLUME_X = (-0.65, -0.15)
    VOLUME_Y = (-0.10,  0.40)
    VOLUME_Z = ( 0.11,  0.40)

    # Fallback default joint angles (deg) when no arm_poses file
    DEFAULT_R1 = [55.53, -139.53, 128.47, -30.04, -236.09, 92.30]
    DEFAULT_R2 = [-50.52, -67.02, -131.03, 244.74, 235.85, -86.97]

    def __init__(self, simulation_app, scene_config, output_dir,
                 num_samples=2000, arm_poses_path=None, cubes_per_pose=None):
        self.simulation_app = simulation_app
        self.output_dir = output_dir
        self.num_samples = num_samples

        # --- Load arm poses ---
        self.arm_poses = None
        self._r1_list = None   # independent list for Robot1
        self._r2_list = None   # independent list for Robot2
        self.cubes_per_pose = None
        if arm_poses_path and os.path.exists(arm_poses_path):
            with open(arm_poses_path) as f:
                data = json.load(f)
            # Two formats:
            #   A) {"robot1": [[...],...], "robot2": [[...],...]}  independent
            #   B) {"poses": [{"robot1": [...], "robot2": [...]},...]} paired
            if "robot1" in data and "robot2" in data and isinstance(data["robot1"][0], list):
                self._r1_list = data["robot1"]
                self._r2_list = data["robot2"]
                print(f"[INFO] Loaded independent lists: "
                      f"Robot1={len(self._r1_list)}, Robot2={len(self._r2_list)}")
            else:
                self.arm_poses = data["poses"]  # paired format (backward compatible)
                print(f"[INFO] Loaded {len(self.arm_poses)} paired pose groups")
            self.cubes_per_pose = cubes_per_pose or max(1, num_samples // 100)
            print(f"[INFO] {self.cubes_per_pose} cube placements per arm pose")
        else:
            print("[INFO] No arm_poses file — using hardcoded default poses")

        # --- Simulation ---
        render_cfg = sim_utils.RenderCfg(
            antialiasing_mode="DLAA",
            enable_dl_denoiser=True,
            dlss_mode=1,
            enable_shadows=False,
        )
        sim_cfg = sim_utils.SimulationCfg(device="cuda:0", render=render_cfg)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.scene = InteractiveScene(scene_config)
        self.sim.reset()

        self._apply_pillar_orientation()

        # Set initial arm pose (random from file, or defaults)
        r1, r2 = self._pick_arm_pose()
        self._write_arm_pose(r1, r2)

        self.camera = self.scene.sensors["viewport_camera"]
        self.device = self.scene.env_origins.device
        self.cube_prim_path = "/World/envs/env_0/Cube"

        os.makedirs(os.path.join(output_dir, "cam_0"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "cam_1"), exist_ok=True)

        # labels: (N, 4) = [x, y, z, pose_index]  when arm_poses is used
        # labels: (N, 3) = [x, y, z]               when no arm_poses
        ncols = 4 if (self.arm_poses or self._r1_list) else 3
        self.labels = np.zeros((num_samples, ncols), dtype=np.float32)

        # Warmup renders
        cam_positions, cam_orientations = get_fixed_cam_poses(self.device)
        self.camera.set_world_poses(cam_positions, cam_orientations, convention="ros")
        self.sim.step(render=True)
        self.camera.update(0, force_recompute=True)
        self.sim.step(render=True)
        self.camera.update(0, force_recompute=True)

    # ------------------------------------------------------------------
    # Pillar orientation fix
    # ------------------------------------------------------------------

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
            inner_path = (
                f"/World/envs/env_{env_idx}/Pillar/"
                "tn__V25_V5xgg2sec0sYY0isSaiJ"
            )
            inner_prim = stage.GetPrimAtPath(inner_path)
            if inner_prim.IsValid():
                xf = UsdGeom.Xformable(inner_prim)
                xf.ClearXformOpOrder()
                op = xf.AddTransformOp()
                op.Set(mat)

    # ------------------------------------------------------------------
    # Arm pose helpers
    # ------------------------------------------------------------------

    def _pick_arm_pose(self, index=-1):
        """Return (r1_deg, r2_deg). Independent lists → random from each."""
        if self._r1_list is not None and self._r2_list is not None:
            i = index if 0 <= index < len(self._r1_list) else np.random.randint(len(self._r1_list))
            j = index if 0 <= index < len(self._r2_list) else np.random.randint(len(self._r2_list))
            return self._r1_list[i], self._r2_list[j]
        if self.arm_poses and 0 <= index < len(self.arm_poses):
            p = self.arm_poses[index]
            return p["robot1"], p["robot2"]
        return self.DEFAULT_R1, self.DEFAULT_R2

    def _write_arm_pose(self, r1_deg, r2_deg):
        """Write joint angles to both robot articulations."""
        targets = {"robot": r1_deg, "robot2": r2_deg}
        for name, deg in targets.items():
            art = self.scene.articulations.get(name)
            if art is None:
                continue
            joint_pos = art.data.default_joint_pos.clone()
            t = torch.tensor(
                [np.deg2rad(a) for a in deg],
                device=joint_pos.device, dtype=joint_pos.dtype,
            )
            joint_pos[0, :6] = t
            art.write_joint_state_to_sim(
                joint_pos, art.data.default_joint_vel.clone()
            )

    # ------------------------------------------------------------------
    # Cube placement
    # ------------------------------------------------------------------

    def _random_cube_position(self):
        x = np.random.uniform(*self.VOLUME_X)
        y = np.random.uniform(*self.VOLUME_Y)
        z = np.random.uniform(*self.VOLUME_Z)
        return np.array([x, y, z], dtype=np.float32)

    def _set_cube_position(self, pos):
        from pxr import Gf, UsdGeom
        import omni.usd

        env_origin = self.scene.env_origins[0].cpu().numpy()
        local = pos - env_origin

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.cube_prim_path)
        if not prim or not prim.IsValid():
            return

        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(float(local[0]), float(local[1]), float(local[2])))
                return

        op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        op.Set(Gf.Vec3d(float(local[0]), float(local[1]), float(local[2])))

    # ------------------------------------------------------------------
    # Camera + render
    # ------------------------------------------------------------------

    def _set_camera_poses(self):
        positions, orientations = get_fixed_cam_poses(self.device)
        self.camera.set_world_poses(positions, orientations, convention="ros")

    # ------------------------------------------------------------------
    # Main capture loop
    # ------------------------------------------------------------------

    def run(self):
        t_start = time.time()

        if self.arm_poses or self._r1_list:
            self._run_with_poses(t_start)
        else:
            self._run_single_pose(t_start)

        elapsed = time.time() - t_start
        print(f"[DONE] {self.num_samples} samples in {elapsed:.1f}s "
              f"({self.num_samples / elapsed:.1f} samples/s)")
        print(f"  → {self.output_dir}")

    def _run_with_poses(self, t_start):
        """Random arm pose combinations per cube placement block."""
        idx = 0
        n = self.num_samples
        cpp = self.cubes_per_pose or 1

        print(f"[INFO] Starting capture: {n} samples (~{cpp} cubes/arm-pose)")

        r1_deg, r2_deg = self._pick_arm_pose()
        self._write_arm_pose(r1_deg, r2_deg)

        for idx in range(n):
            # New arm pose every cpp cubes
            if idx > 0 and idx % cpp == 0:
                r1_deg, r2_deg = self._pick_arm_pose()
                self._write_arm_pose(r1_deg, r2_deg)

            pos = self._random_cube_position()
            self._set_cube_position(pos)
            self._set_camera_poses()

            self.sim.step(render=True)
            self.camera.update(0, force_recompute=True)

            rgb = self.camera.data.output["rgb"]
            img0 = np.flipud(rgb[0].cpu().numpy())
            img1 = np.flipud(rgb[1].cpu().numpy())

            cv2.imwrite(
                os.path.join(self.output_dir, "cam_0", f"{idx:06d}.jpg"),
                cv2.cvtColor(img0, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(self.output_dir, "cam_1", f"{idx:06d}.jpg"),
                cv2.cvtColor(img1, cv2.COLOR_RGB2BGR),
            )
            self.labels[idx] = [pos[0], pos[1], pos[2], -1]

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                eta = (n - idx - 1) / rate if rate > 0 else 0
                print(f"  [{idx + 1:5d}/{n}]  {rate:.1f} samples/s  ETA {eta:.0f}s")

        np.save(os.path.join(self.output_dir, "labels.npy"), self.labels)

    def _run_single_pose(self, t_start):
        """Single fixed pose (no arm_poses file). Original behaviour."""
        n = self.num_samples
        print(f"[INFO] Starting capture: {n} samples (single arm pose)")

        r1, r2 = self._pick_arm_pose(-1)
        self._write_arm_pose(r1, r2)

        for idx in range(n):
            pos = self._random_cube_position()
            self._set_cube_position(pos)
            self._set_camera_poses()

            self.sim.step(render=True)
            self.camera.update(0, force_recompute=True)

            rgb = self.camera.data.output["rgb"]
            img0 = np.flipud(rgb[0].cpu().numpy())
            img1 = np.flipud(rgb[1].cpu().numpy())

            cv2.imwrite(
                os.path.join(self.output_dir, "cam_0", f"{idx:06d}.jpg"),
                cv2.cvtColor(img0, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(self.output_dir, "cam_1", f"{idx:06d}.jpg"),
                cv2.cvtColor(img1, cv2.COLOR_RGB2BGR),
            )
            self.labels[idx] = pos

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                eta = (n - idx - 1) / rate if rate > 0 else 0
                print(f"  [{idx + 1:5d}/{n}]  {rate:.1f} samples/s  ETA {eta:.0f}s")

        np.save(os.path.join(self.output_dir, "labels.npy"), self.labels)

    def close(self):
        self.simulation_app.close()
