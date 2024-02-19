import time
start_time = time.time()
from icecream import ic
import argparse
import numpy as np
import yaml
import cv2
from utils.load_scene import load_scene
from utils.pose_estimate import PoseEstimate
from search_strategy.heuristic_sampling import get_best_pose_estimate, sample_estimates, sample_estimates_gauss
from utils.pointcloud_tools import create_pointclouds, mask_depthmaps, run_icp, draw_registration_result
from utils.camera_conversions import convert_camera_pose_hilla2pyrender
from hilla.geometry.camera import Orientation, PinholeCamera, Pose
from utils.LoFTR import draw_loftr
########## measure execution time of block 1 ##########
print(time.time() - start_time)
#######################################################

class Localizer():
    def __init__(self):
        self.__location = "Japan"

    def run(self, scene, camera, query_pose, guessed_pose):
        ic(self.__location)

        # sample pose estimates and get best one
        sampled_pose_estimates = sample_estimates_gauss(scene, camera, guessed_pose, n=10, bounds = scene.bounds, uncertainty=100)
        estimate_pose = get_best_pose_estimate(query_pose, sampled_pose_estimates)
        #draw estimated pose
        img = np.hstack((query_pose.rgb, estimate_pose.rgb))
        cv2.imshow('estimated pose', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        draw_loftr(query_pose, estimate_pose)
        # create pointclouds for true and best estimate
        masked_depthmap_true_pose, masked_depthmap_estimate_pose = mask_depthmaps(query_pose, estimate_pose) # TODO: what is this doing?
        query_pose = create_pointclouds(query_pose, masked_depthmap_true_pose, camera)
        estimate_pose = create_pointclouds(estimate_pose, masked_depthmap_estimate_pose, camera)

        # ICP to register the pointclouds
        registration = run_icp(estimate_pose, query_pose)
        draw_registration_result(estimate_pose, query_pose, registration.transformation, transform = False)
        draw_registration_result(estimate_pose, query_pose, registration.transformation, transform = True)
        

        predicted_pose = self.registration_to_realworldframe(scene, camera, estimate_pose, registration.transformation)
        
        ic(predicted_pose.camera_pose[0][3])
        ic(query_pose.camera_pose.position.x_coord)
        ic(predicted_pose.camera_pose[1][3])
        ic(query_pose.camera_pose.position.y_coord)
        ic(predicted_pose.camera_pose[2][3])
        ic(query_pose.camera_pose.position.z_coord)

    
    def registration_to_realworldframe(self, scene, camera, best_estimate, predicted_transform):
        source_extrinsic = best_estimate.camera_pose.extrinsic_matrix
        predicted_extrinsic = predicted_transform @ source_extrinsic
        predicted_pose = np.linalg.inv(predicted_extrinsic)
        hilla_pose = Pose.from_camera_in_world(rotation = Orientation.create(predicted_pose[:3, :3]), position = predicted_pose[:3, 3])
        pyrender_pose = convert_camera_pose_hilla2pyrender(hilla_pose)
        result = PoseEstimate.create_from_scene(scene, camera, pyrender_pose, name = 'predicted')
        return result

if __name__ == "__main__":
    new_time = time.time()

    parser = argparse.ArgumentParser(description='Script to export the keypoints for images in a dataset using a base detector')
    #parser.add_argument('-y', '--yaml-config', default='configs/config_export_keypoints.yaml', help='YAML config file')
    parser.add_argument('-i', '--image-dir', default = '/Users/antonia/dev/masterthesis/urban_localization/images/', help='Image directors')
    parser.add_argument('-o', '--obj-dir', default='/Users/antonia/dev/masterthesis/Helsinki3D_2017_OBJ_672496x2/672496a1', help='Directory of the obj files for the 3d model')
    parser.add_argument('-y', '--yaml-config', default='configs/config_synthetic.yaml', help='YAML config file')
    args = parser.parse_args()
    
    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    def initialisation():
        print("Initialising stuff")
        # create pyrender scene object
        scene = load_scene(args.obj_dir)
        # create hilla camera object
        camera = PinholeCamera.from_fov(width=640, height=480, hfov_deg=90)
        # create PoseEstimate object for true pose
        #query_position = [scene.centroid[0], scene.centroid[1] + 50, 60]
        #query_orientation = [-90, -40, 0]
        ic(scene.centroid)
        
        query_pose_hilla = Pose.from_camera_in_world(
            Orientation.from_yaw_pitch_roll(np.radians(config['poses']['query']['orientation'])),
            position=config['poses']['query']['position'],
        )
        query_pose = PoseEstimate.create_from_scene(scene, camera, query_pose_hilla, name = 'query_pose', draw = False)
        # create PoseEstimate object for guessed pose
        guessed_pose_hilla = Pose.from_camera_in_world(
            Orientation.from_yaw_pitch_roll(np.radians(config['poses']['guess']['orientation'])),
            position=config['poses']['guess']['position'],
        )
        guessed_pose = PoseEstimate.create_from_scene(scene, camera, guessed_pose_hilla, name = 'guessed_pose', draw = False)
        return scene, camera, query_pose, guessed_pose

    
    loc = Localizer()
    scene, camera, query_pose, guessed_pose = initialisation()
    
    ########## measure execution time of block 2 ##########
    print(time.time() - new_time)
    new_time = time.time()
    #######################################################

    loc.run(scene, camera, query_pose, guessed_pose)