"""Spatial perception cube capture simulator.

Places a red cube at random positions within a defined 3D volume, captures images
from two fixed cameras, and saves them alongside the cube's world coordinates.

Uses AssetBaseCfg cube (no physics) + USD xform update (no ClearXformOpOrder).
"""

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

    # Cube placement volume (world coordinates, metres)
    VOLUME_X = (-0.55, -0.25)
    VOLUME_Y = (-0.05,  0.35)
    VOLUME_Z = ( 0.11,  0.35)

    def __init__(self, simulation_app, scene_config, output_dir, num_samples=2000):
        self.simulation_app = simulation_app
        self.output_dir = output_dir
        self.num_samples = num_samples

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

        self.camera = self.scene.sensors["viewport_camera"]
        self.device = self.scene.env_origins.device

        self.cube_prim_path = "/World/envs/env_0/Cube"

        os.makedirs(os.path.join(output_dir, "cam_0"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "cam_1"), exist_ok=True)
        self.labels = np.zeros((num_samples, 3), dtype=np.float32)

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
    # Cube placement — update existing translate op only (no clear)
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
        print(f"[INFO] Starting capture: {self.num_samples} samples → {self.output_dir}")
        t_start = time.time()

        for idx in range(self.num_samples):
            self._capture_sample(idx)

            if (idx + 1) % 100 == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                eta = (self.num_samples - idx - 1) / rate if rate > 0 else 0
                print(f"  [{idx + 1:5d}/{self.num_samples}]  "
                      f"{rate:.1f} samples/s  ETA {eta:.0f}s")

        np.save(os.path.join(self.output_dir, "labels.npy"), self.labels)

        elapsed = time.time() - t_start
        print(f"[DONE] {self.num_samples} samples in {elapsed:.1f}s "
              f"({self.num_samples / elapsed:.1f} samples/s)")
        print(f"  → {self.output_dir}")

    def _capture_sample(self, idx):
        pos = self._random_cube_position()
        self._set_cube_position(pos)

        self._set_camera_poses()

        self.sim.step(render=True)
        self.camera.update(0, force_recompute=True)

        rgb = self.camera.data.output["rgb"]
        img0 = rgb[0].cpu().numpy()
        img1 = rgb[1].cpu().numpy()

        # Flip images vertically — camera Y axis maps to world +Z instead of -Z
        img0 = np.flipud(img0)
        img1 = np.flipud(img1)

        cv2.imwrite(
            os.path.join(self.output_dir, "cam_0", f"{idx:06d}.jpg"),
            cv2.cvtColor(img0, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(self.output_dir, "cam_1", f"{idx:06d}.jpg"),
            cv2.cvtColor(img1, cv2.COLOR_RGB2BGR),
        )

        self.labels[idx] = pos

    def close(self):
        self.simulation_app.close()
