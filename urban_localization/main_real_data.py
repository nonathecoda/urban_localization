import numpy as np
import pyrender
import open3d as o3d
import trimesh
from hilla.geometry.camera import PinholeCamera
from pyrender import Scene
from pathlib import Path
from icecream import ic
import tqdm

from utils.load_scene import load_scene, load_obj
from utils.pose_estimate import PoseEstimate
from search_strategy.heuristic_sampling import get_best_pose_estimate, sample_estimates_gauss
from utils.pointcloud_tools import create_pointclouds, mask_depthmaps, run_icp, draw_registration_result
from utils.frame_conversions import convert_camera_pose_hilla2pyrender, registration_to_realworldframe
from hilla.geometry.camera import Orientation, PinholeCamera, Pose

import yaml
with open('/Users/antonia/dev/masterthesis/urban_localization/urban_localization/configs/config_real_data.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

obj_dir = ("/Users/antonia/Downloads/obj_1/textured.obj")
map_dir = "/Users/antonia/dev/masterthesis/Helsinki3D_2017_OBJ_672496x2/"

#### load scene
scene = load_scene(map_dir, tiles = ["672496d2", "672496d1"])

#### create hilla camera object
camera = PinholeCamera.from_fov(width=640, height=480, hfov_deg=90)

### camera pose in scene coordinates
# [6600.        , 4750.        ,   61]
#[160, -40, 0]

### transformation of obj
#scaling_factor=110, translation = [6625.+50, 4750-120., 35], rotation_xyz = 90, -135, 0

#### create query pose
query_pose_hilla = Pose.from_camera_in_world(
    Orientation.from_yaw_pitch_roll(np.radians(config['poses']['query']['orientation'])),
    position=config['poses']['query']['position'],
)
pyrender_pose = convert_camera_pose_hilla2pyrender(query_pose_hilla)
captured_obj = load_obj(obj_dir, pose = pyrender_pose, scaling_factor = config['poses']['captured_transform']['scaling'], translation = config['poses']['captured_transform']['translation'], rotation = config['poses']['captured_transform']['rotation'])
query_pose = PoseEstimate.create_from_obj(captured_obj, camera, query_pose_hilla, name = 'query_pose', draw = True)

#### create guessed pose
guessed_pose_hilla = Pose.from_camera_in_world(
    Orientation.from_yaw_pitch_roll(np.radians(config['poses']['guess']['orientation'])),
    position=config['poses']['guess']['position'],
)
guessed_pose = PoseEstimate.create_from_scene(scene, camera, guessed_pose_hilla, name = 'guessed_pose', draw = False)


sampled_pose_estimates = sample_estimates_gauss(scene, camera, guessed_pose = guessed_pose, n=5, bounds = scene.bounds, uncertainty=[ config['sampling']['sd_guess_x'], config['sampling']['sd_guess_y'] ], pitch = -40)
estimate_pose = get_best_pose_estimate(query_pose, sampled_pose_estimates, draw_loftr_result = True)
#estimate_pose = get_best_pose_estimate(query_pose, [guessed_pose])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(query_pose.depth_map)
ax[0].set_title('query')
ax[1].imshow(estimate_pose.depth_map)
ax[1].set_title('estimate')
#plt.show()

#### create pointclouds
masked_depthmap_true_pose, masked_depthmap_estimate_pose = mask_depthmaps(query_pose, estimate_pose)
estimate_pose = create_pointclouds(estimate_pose, masked_depthmap_estimate_pose, camera)
query_pose = create_pointclouds(query_pose, masked_depthmap_true_pose, camera)

registration = run_icp(estimate_pose,query_pose)

#draw_registration_result(estimate_pose, query_pose, registration.transformation)
pyrender_pose = registration_to_realworldframe(scene, camera, estimate_pose, registration.transformation)
predicted_pose = PoseEstimate.create_from_scene(scene, camera, pyrender_pose, name = 'predicted')
ic(predicted_pose.camera_pose[0][3])
ic(predicted_pose.camera_pose[1][3])
ic(predicted_pose.camera_pose[2][3])