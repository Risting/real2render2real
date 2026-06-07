"""Red cube on pure black background — three orthographic views.

Run on cloud:
    cd .../IsaacLab
    ./isaaclab.sh -p ...debug_cube_views.py
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
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

CUBE_SIZE = 0.05
D435I_INTRINSICS_720 = [910.51, 0, 644.55, 0, 910.20, 369.72, 0, 0, 1]


@configclass
class CubeOnlyCfg(InteractiveSceneCfg):
    cube = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(size=(CUBE_SIZE,)*3,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.15))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, CUBE_SIZE/2)),
    )
    camera = MultiTiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Viewport", height=720, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=D435I_INTRINSICS_720, height=720, width=1280, clipping_range=(0.001, 20)),
        cams_per_env=2,
    )
    # Even multi-directional lighting
    light1 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Light1",
        spawn=sim_utils.CylinderLightCfg(intensity=2000.0, radius=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.8, 0, 0.2)))
    light2 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Light2",
        spawn=sim_utils.CylinderLightCfg(intensity=2000.0, radius=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.8, 0, 0.2)))
    light3 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Light3",
        spawn=sim_utils.CylinderLightCfg(intensity=2000.0, radius=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0.8, 0.2)))
    light4 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Light4",
        spawn=sim_utils.CylinderLightCfg(intensity=2000.0, radius=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, -0.8, 0.2)))
    light5 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Light5",
        spawn=sim_utils.CylinderLightCfg(intensity=2000.0, radius=0.5),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, 1.0)))


cfg = CubeOnlyCfg(num_envs=1, env_spacing=2.0)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cuda:0",
    render=sim_utils.RenderCfg(antialiasing_mode="DLAA", enable_dl_denoiser=True, dlss_mode=1)))
scene = InteractiveScene(cfg)
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
    img = cam.data.output["rgb"][0].cpu().numpy()
    return np.flipud(img)


T = np.array([0.0, 0.0, CUBE_SIZE/2])
out = "/root/gpufree-data/spatial_perception_dataset/cube_views"
os.makedirs(out, exist_ok=True)
d = 0.20  # camera distance

for name, eye in [("front", (0, -d, CUBE_SIZE/2)), ("side", (-d, 0, CUBE_SIZE/2)), ("top", (0, 0, d))]:
    cv2.imwrite(f"{out}/{name}.jpg", cv2.cvtColor(capture(np.array(eye), T), cv2.COLOR_RGB2BGR))
    print(f"  Saved {name}.jpg")

print(f"\nDone → {out}")
simulation_app.close()
