"""One shape three-view: front / side / top. Black background.

Usage:
    ./isaaclab.sh -p ...debug_cube_views.py --shape cube
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--shape", type=str, default="cube", choices=["cube","cuboid","cylinder","sphere"])
parser.add_argument("--output_dir", type=str, default="/root/gpufree-data/spatial_perception_dataset")
args_cli = parser.parse_args()
args_cli.headless = True; args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os, cv2, numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiTiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_matrix

D435I = [910.51, 0, 644.55, 0, 910.20, 369.72, 0, 0, 1]

SHAPE_CFG = {
    "cube":     sim_utils.CuboidCfg(size=(0.05,)*3, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9,0.15,0.15))),
    "cuboid":   sim_utils.CuboidCfg(size=(0.08,0.04,0.04), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15,0.7,0.15))),
    "cylinder": sim_utils.CylinderCfg(radius=0.025, height=0.08, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15,0.4,0.9))),
    "sphere":   sim_utils.SphereCfg(radius=0.035, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9,0.7,0.1))),
}


@configclass
class BaseCfg(InteractiveSceneCfg):
    camera = MultiTiledCameraCfg(prim_path="{ENV_REGEX_NS}/Viewport", height=720, width=1280,
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=D435I, height=720, width=1280, clipping_range=(0.001, 20)), cams_per_env=2)
    l1 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/L1", spawn=sim_utils.CylinderLightCfg(intensity=2000, radius=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, 0, 0.2)))
    l2 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/L2", spawn=sim_utils.CylinderLightCfg(intensity=2000, radius=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.8, 0, 0.2)))
    l3 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/L3", spawn=sim_utils.CylinderLightCfg(intensity=2000, radius=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0.8, 0.2)))
    l4 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/L4", spawn=sim_utils.CylinderLightCfg(intensity=2000, radius=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, -0.8, 0.2)))
    l5 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/L5", spawn=sim_utils.CylinderLightCfg(intensity=2000, radius=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 1.0)))


cfg = BaseCfg(num_envs=1, env_spacing=2.0)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0",
    render=sim_utils.RenderCfg(antialiasing_mode="DLAA", enable_dl_denoiser=True, dlss_mode=1)))
scene = InteractiveScene(cfg)

# Spawn shape manually
from pxr import UsdGeom, Gf; import omni.usd
stage = omni.usd.get_context().get_stage()
spawn_cfg = SHAPE_CFG[args_cli.shape]
spawn_cfg.func("/World/envs/env_0/Obj", spawn_cfg)

sim.reset()
cam = scene.sensors["camera"]


def look_at_quat(eye, target):
    fwd = target - eye; fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0., 0., 1.])
    if abs(np.dot(fwd, up)) > 0.999: up = np.array([0., 1., 0.])
    right = np.cross(up, fwd); right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    return quat_from_matrix(torch.from_numpy(np.column_stack([right, down, fwd])).float())


def capture(eye, target):
    eye_t = torch.tensor(eye, dtype=torch.float32)
    quat = look_at_quat(eye, target)
    cam.set_world_poses(torch.stack([eye_t, eye_t]), torch.stack([quat, quat]), convention="ros")
    sim.step(render=True); cam.update(0, force_recompute=True)
    sim.step(render=True); cam.update(0, force_recompute=True)
    return np.flipud(cam.data.output["rgb"][0].cpu().numpy())


out = f"{args_cli.output_dir}/{args_cli.shape}"
os.makedirs(out, exist_ok=True)
T = np.array([0.0, 0.0, 0.025])

for vname, eye in [("front", (0, -0.20, 0.025)), ("side", (-0.20, 0, 0.025)), ("top", (0, 0, 0.20))]:
    cv2.imwrite(f"{out}/{vname}.jpg", cv2.cvtColor(capture(np.array(eye), T), cv2.COLOR_RGB2BGR))
    print(f"  {out}/{vname}.jpg")

print("Done")
simulation_app.close()
