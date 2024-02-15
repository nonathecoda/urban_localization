import time
start_time = time.time()
from icecream import ic
import argparse
import numpy as np
import yaml

from utils.load_scene import load_scene
from utils.pose_estimate import PoseEstimate
from search_strategy.heuristic_sampling import get_best_pose_estimate, sample_estimates
from utils.pointcloud_tools import create_pointclouds, mask_depthmaps
from hilla.geometry.camera import Orientation, PinholeCamera, Pose
from hilla.common import RectangleSize

########## measure execution time of block 1 ##########
print(time.time() - start_time)
#######################################################

class Localizer():
    def __init__(self):
        self.__location = "Japan"

    def run(self, scene, camera, true_pose, guessed_pose):
        ic(self.__location)

        # sample pose estimates and get best one
        sampled_pose_estimates = sample_estimates(scene, camera, guessed_pose, n=10)
        estimate_pose = get_best_pose_estimate(true_pose, sampled_pose_estimates)
        
        # create pointclouds for true and best estimate
        masked_depthmap_true_pose, masked_depthmap_estimate_pose = mask_depthmaps(true_pose, estimate_pose) # TODO: what is this doing?
        create_pointclouds(true_pose, masked_depthmap_true_pose, camera)
        create_pointclouds(estimate_pose, masked_depthmap_estimate_pose, camera)

       
         


if __name__ == "__main__":
    new_time = time.time()

    parser = argparse.ArgumentParser(description='Script to export the keypoints for images in a dataset using a base detector')
    #parser.add_argument('-y', '--yaml-config', default='configs/config_export_keypoints.yaml', help='YAML config file')
    parser.add_argument('-i', '--image-dir', default = '/Users/antonia/dev/masterthesis/urban_localization/images/', help='Image directors')
    parser.add_argument('-o', '--obj-dir', default='/Users/antonia/dev/masterthesis/Helsinki3D_2017_OBJ_672496x2/672496a1', help='Directory of the obj files for the 3d model')
    parser.add_argument('y', '--yaml-config', default='configs/config_real_data.yaml', help='YAML config file')
    parser.add_argument('-q', '--query-image', default='images/query-py', help='Path to query image (before split)')
    args = parser.parse_args()

    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    def initialisation(config, args):
        print("Initialising stuff")
        # create pyrender scene object
        scene = load_scene(args.obj_dir)
        # create hilla camera object
        intrinsics_left = np.load(config['camera']['intrinsics_left'])
        camera_left = PinholeCamera()
        camera_left._resolution = RectangleSize.from_width_height(config['camera']['width'], config['camera']['height'])
        camera_left.fx = intrinsics_left[0, 0]
        camera_left.fy = intrinsics_left[1, 1]
        camera_left.px = intrinsics_left[0, 2]
        camera_left.py = intrinsics_left[1, 2]
        # create PoseEstimate object for true pose
        query_position = [scene.centroid[0], scene.centroid[1] + 50, 60] # TODO: ADD true position
        query_orientation = [-90, -40, 0] # TODO: ADD true orientation
        query_pose_hilla = Pose.from_camera_in_world(
            Orientation.from_yaw_pitch_roll(np.radians(query_orientation)),
            position=query_position,
        )
        query_pose = PoseEstimate.create_from_image(scene, camera_left, query_pose_hilla, name = 'query_pose', config = config, args = args)
        # create PoseEstimate object for guessed pose
        guessed_position = np.add(query_position, [50, -45, 30])
        guessed_orientation = np.add(query_orientation, [10, 0, 0])
        guessed_pose_hilla = Pose.from_camera_in_world(
            Orientation.from_yaw_pitch_roll(np.radians(guessed_orientation)),
            position=guessed_position,
        )
        guessed_pose = PoseEstimate.create_from_scene(scene, camera_left, guessed_pose_hilla, name = 'guessed_pose')
        return scene, camera_left, query_pose, guessed_pose

    loc = Localizer()
    scene, camera_left, query_pose, guessed_pose = initialisation(config, args)
    
    ########## measure execution time of block 2 ##########
    print(time.time() - new_time)
    new_time = time.time()
    #######################################################

    loc.run(scene, camera_left, query_pose, guessed_pose)