"""Single sample: same cube, three orthogonal camera views (front/side/top).

Run on cloud:
    cd /root/gpufree-data/r2r2r/dependencies/IsaacLab
    ./isaaclab.sh -p /root/gpufree-data/r2r2r/real2render2real/isaaclab_viser/spatial_perception/debug_cube_views.py
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, cv2, numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_from_matrix
from real2render2real.isaaclab_viser.configs.scene_configs.spatial_perception_cfg import SpatialPerceptionCfg

# Setup scene (single camera, we reposition it each shot)
cfg = SpatialPerceptionCfg(num_envs=1, env_spacing=2.0)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    device="cuda:0",
    render=sim_utils.RenderCfg(antialiasing_mode="DLAA", enable_dl_denoiser=True, dlss_mode=1, enable_shadows=False),
))
scene = InteractiveScene(cfg)
sim.reset()

# Fix pillar + set arms
from pxr import Gf, UsdGeom
import omni.usd
stage = omni.usd.get_context().get_stage()
mat = Gf.Matrix4d(
     0.50545858, -0.01564236, -0.86270907,  0,
     0.86144504, -0.04790288,  0.50558655,  0,
    -0.04923481, -0.99872951, -0.01073789,  0,
    -576.34225775, 710.48252687, 523.69519926, 1,
)
for env_idx in range(scene.num_envs):
    inner = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/Pillar/tn__V25_V5xgg2sec0sYY0isSaiJ")
    if inner.IsValid():
        xf = UsdGeom.Xformable(inner); xf.ClearXformOpOrder(); xf.AddTransformOp().Set(mat)

for name in scene.articulations:
    art = scene.articulations[name]
    art.write_joint_state_to_sim(art.data.default_joint_pos, art.data.default_joint_vel)

cube_prim = "/World/envs/env_0/Cube"
cam = scene.sensors["viewport_camera"]

# Place cube at a known position
CUBE_POS = np.array([-0.40, 0.15, 0.25], dtype=np.float32)
env_origin = scene.env_origins[0].cpu().numpy()
local = CUBE_POS - env_origin
prim = stage.GetPrimAtPath(cube_prim)
if prim and prim.IsValid():
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(float(local[0]), float(local[1]), float(local[2])))
            break


def look_at_quat(eye, target):
    fwd = target - eye; fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0., 0., 1.])
    if abs(np.dot(fwd, up)) > 0.999: up = np.array([0., 1., 0.])
    right = np.cross(up, fwd); right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.column_stack([right, down, fwd])
    return quat_from_matrix(torch.from_numpy(R).float())


# Three camera views
T = np.array([-0.40, 0.15, 0.25])  # look at cube
views = {
    "front": np.array([-0.40, -1.50, 0.25]),
    "side":  np.array([-1.80,  0.15, 0.25]),
    "top":   np.array([-0.40,  0.15, 1.50]),
}

out_dir = "/root/gpufree-data/spatial_perception_dataset/cube_views"
os.makedirs(out_dir, exist_ok=True)

for name, eye in views.items():
    # Workaround: use 2 cameras, set both to same pose, take cam_0
    eye_t = torch.tensor(eye, dtype=torch.float32)
    quat = look_at_quat(eye, T)
    positions = torch.stack([eye_t, eye_t], dim=0)
    orientations = torch.stack([quat, quat], dim=0)
    cam.set_world_poses(positions, orientations, convention="ros")
    sim.step(render=True)
    cam.update(0, force_recompute=True)
    sim.step(render=True)
    cam.update(0, force_recompute=True)
    img = cam.data.output["rgb"][0].cpu().numpy()
    img = np.flipud(img)
    cv2.imwrite(os.path.join(out_dir, f"{name}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"  Saved {name}.jpg")

print(f"\nDone. Cube at {CUBE_POS}. Images in {out_dir}")
simulation_app.close()
