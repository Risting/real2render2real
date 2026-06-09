"""Spatial perception capture simulator.

Random arm poses + random object shapes + random positions.
Shapes: cube, cuboid, cylinder, sphere — visibility toggled per sample.
"""

import json, os, time, cv2, numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from real2render2real.isaaclab_viser.spatial_perception.camera_extrinsics import get_fixed_cam_poses


class SpatialCubeCapture:

    VOLUME_X = (-0.65, -0.15)
    VOLUME_Y = (-0.10,  0.40)
    VOLUME_Z = ( 0.11,  0.40)

    DEFAULT_R1 = [55.53, -139.53, 128.47, -30.04, -236.09, 92.30]
    DEFAULT_R2 = [-50.52, -67.02, -131.03, 244.74, 235.85, -86.97]

    SHAPE_LIST = ["cube", "cuboid", "cylinder", "sphere"]

    def __init__(self, simulation_app, scene_config, output_dir,
                 num_samples=2000, arm_poses_path=None, cubes_per_pose=None, shapes=None):
        self.simulation_app = simulation_app
        self.output_dir = output_dir
        self.num_samples = num_samples
        self._shape_filter = shapes or self.SHAPE_LIST  # e.g. ["cube"] or ["cube","sphere"]

        # --- Load arm poses ---
        self.arm_poses = None
        self._r1_list = self._r2_list = None
        self.cubes_per_pose = None
        if arm_poses_path and os.path.exists(arm_poses_path):
            with open(arm_poses_path) as f: data = json.load(f)
            if "robot1" in data and isinstance(data["robot1"][0], list):
                self._r1_list = data["robot1"]; self._r2_list = data["robot2"]
                print(f"[INFO] Loaded independent lists: R1={len(self._r1_list)}, R2={len(self._r2_list)}")
            else:
                self.arm_poses = data["poses"]
                print(f"[INFO] Loaded {len(self.arm_poses)} paired pose groups")
            self.cubes_per_pose = cubes_per_pose or max(1, num_samples // 100)
        else:
            print("[INFO] No arm_poses file — using hardcoded defaults")

        # --- Simulation ---
        render_cfg = sim_utils.RenderCfg(antialiasing_mode="DLAA", enable_dl_denoiser=True, dlss_mode=1, enable_shadows=False)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0", render=render_cfg))
        self.scene = InteractiveScene(scene_config)
        self.sim.reset()

        self._apply_pillar_orientation()
        r1, r2 = self._pick_arm_pose()
        self._write_arm_pose(r1, r2)

        self.camera = self.scene.sensors["viewport_camera"]
        self.device = self.scene.env_origins.device

        self.shape_paths = {s: f"/World/envs/env_0/{s.capitalize()}" for s in self.SHAPE_LIST}
        self._current_shape = "cube"

        os.makedirs(os.path.join(output_dir, "cam_0"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "cam_1"), exist_ok=True)

        has_poses = bool(self.arm_poses or self._r1_list)
        ncols = 5 if has_poses else 4  # x,y,z,shape_id[,pose_index]
        self.labels = np.zeros((num_samples, ncols), dtype=np.float32)

        # Warmup
        cam_pos, cam_ori = get_fixed_cam_poses(self.device)
        self.camera.set_world_poses(cam_pos, cam_ori, convention="ros")
        self.sim.step(render=True); self.camera.update(0, force_recompute=True)
        self.sim.step(render=True); self.camera.update(0, force_recompute=True)

    # ------------------------------------------------------------------ Pillar

    def _apply_pillar_orientation(self):
        from pxr import Gf, UsdGeom; import omni.usd
        stage = omni.usd.get_context().get_stage()
        mat = Gf.Matrix4d(
             0.5054585783556504, -0.01564235936930266, -0.862709071564713,   0,
             0.8614450409941848, -0.0479028782148914,   0.5055865461078429,  0,
            -0.04923481403654982, -0.9987295083515333, -0.010737887813508283, 0,
            -576.3422577523722,    710.4825268723347,    523.6951992568527,   1)
        for env_idx in range(self.scene.num_envs):
            inner = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/Pillar/tn__V25_V5xgg2sec0sYY0isSaiJ")
            if inner.IsValid():
                xf = UsdGeom.Xformable(inner); xf.ClearXformOpOrder(); xf.AddTransformOp().Set(mat)

    # ------------------------------------------------------------------ Arm poses

    def _pick_arm_pose(self, index=-1):
        if self._r1_list:
            i = index if 0 <= index < len(self._r1_list) else np.random.randint(len(self._r1_list))
            j = index if 0 <= index < len(self._r2_list) else np.random.randint(len(self._r2_list))
            return self._r1_list[i], self._r2_list[j]
        if self.arm_poses and 0 <= index < len(self.arm_poses):
            p = self.arm_poses[index]; return p["robot1"], p["robot2"]
        return self.DEFAULT_R1, self.DEFAULT_R2

    def _write_arm_pose(self, r1_deg, r2_deg):
        for name, deg in [("robot", r1_deg), ("robot2", r2_deg)]:
            art = self.scene.articulations.get(name)
            if art is None: continue
            jp = art.data.default_joint_pos.clone()
            jp[0, :6] = torch.tensor([np.deg2rad(a) for a in deg], device=jp.device, dtype=jp.dtype)
            art.write_joint_state_to_sim(jp, art.data.default_joint_vel.clone())

    # ------------------------------------------------------------------ Shape

    def _pick_shape(self):
        lst = self._shape_filter
        s = lst[np.random.randint(len(lst))]
        return s, self.SHAPE_LIST.index(s)

    def _show_shape(self, shape_name):
        from pxr import UsdGeom; import omni.usd
        stage = omni.usd.get_context().get_stage()
        for name, path in self.shape_paths.items():
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                attr = prim.GetAttribute("visibility")
                if not attr: attr = prim.CreateAttribute("visibility", UsdGeom.Tokens.visibility)
                attr.Set("inherited" if name == shape_name else "invisible")
        self._current_shape = shape_name

    # ------------------------------------------------------------------ Position

    def _random_position(self):
        return np.array([np.random.uniform(*r) for r in (self.VOLUME_X, self.VOLUME_Y, self.VOLUME_Z)], dtype=np.float32)

    def _set_position(self, pos):
        from pxr import Gf, UsdGeom; import omni.usd
        local = pos - self.scene.env_origins[0].cpu().numpy()
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(self.shape_paths[self._current_shape])
        if not prim or not prim.IsValid(): return
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(float(local[0]), float(local[1]), float(local[2]))); return
        op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        op.Set(Gf.Vec3d(float(local[0]), float(local[1]), float(local[2])))

    # ------------------------------------------------------------------ Camera

    def _set_camera_poses(self):
        p, o = get_fixed_cam_poses(self.device)
        self.camera.set_world_poses(p, o, convention="ros")

    # ------------------------------------------------------------------ Capture

    def run(self):
        t0 = time.time()
        if self.arm_poses or self._r1_list:
            self._run_multi_pose(t0)
        else:
            self._run_single_pose(t0)
        print(f"[DONE] {self.num_samples} samples in {time.time()-t0:.1f}s → {self.output_dir}")

    def _do_one(self, idx, shape_id):
        pos = self._random_position()
        self._set_position(pos)
        self._set_camera_poses()
        self.sim.step(render=True); self.camera.update(0, force_recompute=True)
        rgb = self.camera.data.output["rgb"]
        cv2.imwrite(f"{self.output_dir}/cam_0/{idx:06d}.jpg", cv2.cvtColor(np.flipud(rgb[0].cpu().numpy()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{self.output_dir}/cam_1/{idx:06d}.jpg", cv2.cvtColor(np.flipud(rgb[1].cpu().numpy()), cv2.COLOR_RGB2BGR))
        if self.labels.shape[1] == 5:
            self.labels[idx] = [pos[0], pos[1], pos[2], shape_id, -1]
        else:
            self.labels[idx] = [pos[0], pos[1], pos[2], shape_id]

    def _run_multi_pose(self, t0):
        n, cpp = self.num_samples, self.cubes_per_pose or 1
        print(f"[INFO] Starting: {n} samples (~{cpp} cubes/pose)")
        r1, r2 = self._pick_arm_pose()
        self._write_arm_pose(r1, r2)
        for idx in range(n):
            if idx > 0 and idx % cpp == 0:
                r1, r2 = self._pick_arm_pose(); self._write_arm_pose(r1, r2)
            sname, sid = self._pick_shape(); self._show_shape(sname)
            self._do_one(idx, sid)
            if (idx+1) % 100 == 0:
                e = time.time()-t0; print(f"  [{idx+1:5d}/{n}]  {(idx+1)/e:.1f} samples/s  ETA {(n-idx-1)/((idx+1)/e):.0f}s")
        np.save(f"{self.output_dir}/labels.npy", self.labels)

    def _run_single_pose(self, t0):
        n = self.num_samples
        print(f"[INFO] Starting: {n} samples (single arm pose)")
        r1, r2 = self._pick_arm_pose(); self._write_arm_pose(r1, r2)
        for idx in range(n):
            sname, sid = self._pick_shape(); self._show_shape(sname)
            self._do_one(idx, sid)
            if (idx+1) % 100 == 0:
                e = time.time()-t0; print(f"  [{idx+1:5d}/{n}]  {(idx+1)/e:.1f} samples/s  ETA {(n-idx-1)/((idx+1)/e):.0f}s")
        np.save(f"{self.output_dir}/labels.npy", self.labels)

    def close(self):
        self.simulation_app.close()
