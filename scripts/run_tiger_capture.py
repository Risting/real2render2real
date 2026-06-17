"""Capture tiger at random positions — 10 test samples.

Run on cloud:
    ./isaaclab.sh -p ...run_tiger_capture.py --num_samples 10
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_samples", type=int, default=10)
parser.add_argument("--output_dir", type=str,
                    default="/root/gpufree-data/spatial_perception_dataset_v2/tiger")
args_cli = parser.parse_args()
args = args_cli  # shorthand
args_cli.headless = True; args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json, os, cv2, numpy as np, torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from real2render2real.isaaclab_viser.spatial_perception.camera_extrinsics import get_fixed_cam_poses
from real2render2real.isaaclab_viser.configs.scene_configs.spatial_perception_cfg import SpatialPerceptionCfg

VOL_X = (-0.65, -0.15); VOL_Y = (-0.10, 0.40); VOL_Z = (0.10, 0.30)

TIGER_USD = os.path.abspath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "data", "assets", "object_scans", "tiger", "tiger_new.usd"))

cfg = SpatialPerceptionCfg(num_envs=1, env_spacing=2.0)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    device="cuda:0",
    render=sim_utils.RenderCfg(antialiasing_mode="DLAA", enable_dl_denoiser=True, dlss_mode=1, enable_shadows=False)))
scene = InteractiveScene(cfg)
sim.reset()

from pxr import Gf, UsdGeom, UsdPhysics; import omni.usd
stage = omni.usd.get_context().get_stage()
# Pillar
mat = Gf.Matrix4d(
     0.5054585783556504, -0.01564235936930266, -0.862709071564713,   0,
     0.8614450409941848, -0.0479028782148914,   0.5055865461078429,  0,
    -0.04923481403654982, -0.9987295083515333, -0.010737887813508283, 0,
    -576.3422577523722,    710.4825268723347,    523.6951992568527,   1)
for env_idx in range(scene.num_envs):
    inner = stage.GetPrimAtPath(f"/World/envs/env_{env_idx}/Pillar/tn__V25_V5xgg2sec0sYY0isSaiJ")
    if inner.IsValid():
        xf = UsdGeom.Xformable(inner); xf.ClearXformOpOrder(); xf.AddTransformOp().Set(mat)

# Hide shapes
for name in ["Cube", "Cuboid", "Cylinder", "Sphere"]:
    prim = stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
    if prim and prim.IsValid():
        attr = prim.GetAttribute("visibility")
        if not attr: attr = prim.CreateAttribute("visibility", UsdGeom.Tokens.visibility)
        attr.Set("invisible")

# Spawn tiger
tiger_path = "/World/envs/env_0/Tiger"
if not stage.GetPrimAtPath(tiger_path).IsValid():
    tiger_prim = UsdGeom.Xform.Define(stage, tiger_path).GetPrim()
    tiger_prim.GetReferences().AddReference(TIGER_USD)
    print(f"[INFO] Tiger spawned from {TIGER_USD}")

# Arm poses
R1_DEF = [55.53, -139.53, 128.47, -30.04, -236.09, 92.30]
R2_DEF = [-50.52, -67.02, -131.03, 244.74, 235.85, -86.97]
arm_json = "/root/gpufree-data/r2r2r/spatial_perception/arm_poses.json"
if os.path.exists(arm_json):
    ap = json.load(open(arm_json))
    R1_LIST, R2_LIST = ap["robot1"], ap["robot2"]
    print(f"[INFO] Arm poses: R1={len(R1_LIST)}, R2={len(R2_LIST)}")
else:
    R1_LIST, R2_LIST = [R1_DEF], [R2_DEF]

def set_arm_pose():
    r1 = R1_LIST[np.random.randint(len(R1_LIST))]
    r2 = R2_LIST[np.random.randint(len(R2_LIST))]
    for art_name, deg in [("robot", r1), ("robot2", r2)]:
        art = scene.articulations.get(art_name)
        if art is None: continue
        jp = art.data.default_joint_pos.clone()
        jp[0,:6] = torch.tensor([np.deg2rad(a) for a in deg], device=jp.device, dtype=jp.dtype)
        art.write_joint_state_to_sim(jp, art.data.default_joint_vel.clone())

cam = scene.sensors["viewport_camera"]
device = scene.env_origins.device
env_origin = scene.env_origins[0].cpu().numpy()

for d in ["cam_0","cam_1","depth_0","depth_1"]:
    os.makedirs(f"{args.output_dir}/{d}", exist_ok=True)
labels = np.zeros((args.num_samples, 3), dtype=np.float32)

pos, ori = get_fixed_cam_poses(device)
cam.set_world_poses(pos, ori, convention="ros")
sim.step(render=True); cam.update(0, force_recompute=True)
sim.step(render=True); cam.update(0, force_recompute=True)

print(f"[INFO] Capturing {args.num_samples} tiger samples → {args.output_dir}")
for idx in range(args.num_samples):
    set_arm_pose()

    p = np.array([np.random.uniform(*VOL_X), np.random.uniform(*VOL_Y), np.random.uniform(*VOL_Z)], dtype=np.float32)
    labels[idx] = p
    local = p - env_origin
    prim = stage.GetPrimAtPath(tiger_path)
    if prim and prim.IsValid():
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(float(local[0]), float(local[1]), float(local[2]))); break
        else:
            xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(float(local[0]), float(local[1]), float(local[2])))

    pos, ori = get_fixed_cam_poses(device)
    cam.set_world_poses(pos, ori, convention="ros")
    sim.step(render=True); cam.update(0, force_recompute=True)

    rgb = cam.data.output["rgb"]
    depth = cam.data.output.get("depth")
    cv2.imwrite(f"{args.output_dir}/cam_0/{idx:06d}.jpg",
                cv2.cvtColor(np.flipud(rgb[0].cpu().numpy()), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{args.output_dir}/cam_1/{idx:06d}.jpg",
                cv2.cvtColor(np.flipud(rgb[1].cpu().numpy()), cv2.COLOR_RGB2BGR))
    if depth is not None:
        for ci, dn in [(0,"depth_0"),(1,"depth_1")]:
            d = np.clip(np.flipud(depth[ci].cpu().numpy().squeeze()), 0, 5) / 5 * 255
            cv2.imwrite(f"{args.output_dir}/{dn}/{idx:06d}.png", d.astype(np.uint8))
    print(f"  [{idx+1}/{args.num_samples}]  pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")

np.save(f"{args.output_dir}/labels.npy", labels)
print(f"\nDone → {args.output_dir}  labels: {labels.shape}")
simulation_app.close()
