import time
start_time = time.time()
import os
from icecream import ic
import argparse
import numpy as np
import yaml
import cv2
from utils.load_scene import load_scene, choose_random_map, change_poses, load_entire_scene, load_obj
from utils.pose_estimate import PoseEstimate
from search_strategy.heuristic_sampling import get_best_pose_estimate, sample_estimates_gauss
from utils.pointcloud_tools import create_pointclouds, mask_depthmaps, run_icp, draw_registration_result
from utils.frame_conversions import convert_camera_pose_hilla2pyrender, rotation_matrix_to_yaw_pitch_roll, registration_to_realworldframe
from hilla.geometry.camera import Orientation, PinholeCamera, Pose
from utils.LoFTR import draw_loftr
import open3d as o3d

class Localizer():
    def __init__(self, scene, camera, query_pose, guessed_pose, config, args, index =0):
        self.__scene = scene
        self.__camera = camera
        self.__query_pose = query_pose
        self.__guessed_pose = guessed_pose
        self.__predicted_pose = None
        self.__estimate_pose = None
        self.__config = config
        self.__args = args
        self.__index = index

        print("Localizer initialised:")
        ic(config['poses']['query']['position'])
        ic(config['poses']['query']['orientation'])
        ic(config['poses']['guess']['position'])
        ic(config['poses']['guess']['orientation'])

    def run(self, time_stamp):
        
        # sample pose estimates and get best one
        sampled_pose_estimates = sample_estimates_gauss(self.__scene, self.__camera, self.__guessed_pose, n=self.__config['sampling']['number_samples'], bounds = self.__scene.bounds, uncertainty=[self.__config['sampling']['sd_guess_x'],self.__config['sampling']['sd_guess_y'] ])
        self.__estimate_pose = get_best_pose_estimate(self.__query_pose, sampled_pose_estimates, draw_loftr_result = False)
        '''
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(self.__query_pose.depth_map)
        ax[0].set_title('query')
        ax[0].set_axis_off()
        ax[1].imshow(self.__estimate_pose.depth_map)
        ax[1].set_title('estimate')
        ax[1].set_axis_off()
        plt.show()
        '''
        #draw estimated pose
        #img = np.hstack((self.__query_pose.rgb, self.__estimate_pose.rgb))
        #cv2.imshow('estimated pose', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #draw_loftr(self.__query_pose, self.__estimate_pose)
        # create pointclouds for true and best estimate
        masked_depthmap_true_pose, masked_depthmap_estimate_pose = mask_depthmaps(self.__query_pose, self.__estimate_pose) # TODO: what is this doing?
        
        self.__query_pose = create_pointclouds(self.__query_pose, masked_depthmap_true_pose, self.__camera)
        self.__estimate_pose = create_pointclouds(self.__estimate_pose, masked_depthmap_estimate_pose, self.__camera)

        # ICP to register the pointclouds
        registration = run_icp(self.__estimate_pose, self.__query_pose)
#        draw_registration_result(self.__estimate_pose, self.__query_pose, registration.transformation, transform=False)
 #       draw_registration_result(self.__estimate_pose, self.__query_pose, registration.transformation, transform=True)
        

        pyrender_pose = registration_to_realworldframe(self.__scene, self.__camera, self.__estimate_pose, registration.transformation)
        self.__predicted_pose = PoseEstimate.create_from_scene(scene, camera, pyrender_pose, name = 'predicted')
        ic(self.__predicted_pose.camera_pose[0][3])
        ic(self.__predicted_pose.camera_pose[1][3])
        ic(self.__predicted_pose.camera_pose[2][3])

        self.save_results(self.__predicted_pose, time_stamp) 


    
    def save_results(self, predicted_pose, time_stamp):
        # calculate the error between the true and predicted pose
        
        iteration_time = float(time.time() - time_stamp)
        predicted_pose_yaw, predicted_pose_pitch, predicted_pose_roll = rotation_matrix_to_yaw_pitch_roll(R = predicted_pose.camera_pose[0:3, 0:3])
        ###save results to yaml file
        
        predicted_position = [float(predicted_pose.camera_pose[0][3]), float(predicted_pose.camera_pose[1][3]), float(predicted_pose.camera_pose[2][3])]
        predicted_orientation = [float(predicted_pose_yaw), float(predicted_pose_pitch), float(predicted_pose_roll)]
        query_position = [float(self.__config['poses']['query']['position'][0]), float(self.__config['poses']['query']['position'][1]), float(self.__config['poses']['query']['position'][2])]
        query_orientation = [float(self.__config['poses']['query']['orientation'][0]), float(self.__config['poses']['query']['orientation'][1]), float(self.__config['poses']['query']['orientation'][2])]
        guessed_position = [float(self.__config['poses']['guess']['position'][0]), float(self.__config['poses']['guess']['position'][1]), float(self.__config['poses']['guess']['position'][2])]
        guessed_orientation = [float(self.__config['poses']['guess']['orientation'][0]), float(self.__config['poses']['guess']['orientation'][1]), float(self.__config['poses']['guess']['orientation'][2])]

        result = {  'predicted_position': predicted_position,
                    'predicted_orientation': predicted_orientation,
                    'query_position': query_position,
                    'query_orientation': query_orientation,
                    'guessed_position': guessed_position,
                    'guessed_orientation': guessed_orientation,
                    'execution_time': iteration_time,
                  }
       
        folder_directory = self.__args.save_dir + '/' + str(self.__config['results']['folder'])
        if not os.path.exists(folder_directory):
            os.makedirs(folder_directory)
        
        save_dir = folder_directory + str(self.__index) + '.yaml'
        with open(save_dir, 'w') as file:
            yaml.dump(result, file)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to export the keypoints for images in a dataset using a base detector')
    #parser.add_argument('-y', '--yaml-config', default='configs/config_export_keypoints.yaml', help='YAML config file')
    parser.add_argument('-i', '--image-dir', default = '/Users/antonia/dev/masterthesis/urban_localization/images/', help='Image directors')
    parser.add_argument('-m', '--map-dir', default='/Users/antonia/dev/masterthesis/Helsinki3D_2017_OBJ_672496x2/', help='Directory of the obj files for the 3d model')
    parser.add_argument('-o', '--obj-dir', default="/Users/antonia/Downloads/obj_1/textured.obj", help='Directory of the obj files for the 3d model')
    parser.add_argument('-y', '--yaml-config', default='urban_localization/configs/config_real_data.yaml', help='YAML config file')
    parser.add_argument('-s', '--save-dir', default='/Users/antonia/dev/masterthesis/urban_localization/urban_localization/results_field_exp', help='Directory to save the results in YAML')
    args = parser.parse_args()
    
    with open(args.yaml_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    def initialisation(config, scene = None):
        print("Initialising stuff")
        # create pyrender scene object
        
        if config['experiment']['random_pose'] == True:
            config = change_poses(config, scene, change_query = False)
        # create hilla camera object
        camera = PinholeCamera.from_fov(width=640, height=480, hfov_deg=90)
        #### create query pose
        query_pose_hilla = Pose.from_camera_in_world(
            Orientation.from_yaw_pitch_roll(np.radians(config['poses']['query']['orientation'])),
            position=config['poses']['query']['position'],
        )
        pyrender_pose = convert_camera_pose_hilla2pyrender(query_pose_hilla)
        captured_obj = load_obj(args.obj_dir, pose = pyrender_pose, scaling_factor = config['poses']['captured_transform']['scaling'], translation = config['poses']['captured_transform']['translation'], rotation = config['poses']['captured_transform']['rotation'])
        query_pose = PoseEstimate.create_from_obj(captured_obj, camera, query_pose_hilla, name = 'query_pose', draw = False)

        # create PoseEstimate object for guessed pose
        #### create guessed pose
        guessed_pose_hilla = Pose.from_camera_in_world(
            Orientation.from_yaw_pitch_roll(np.radians(config['poses']['guess']['orientation'])),
            position=config['poses']['guess']['position'],
        )
        guessed_pose = PoseEstimate.create_from_scene(scene, camera, guessed_pose_hilla, name = 'guessed_pose', draw = False)
        return scene, camera, query_pose, guessed_pose, config

    #### load scene
    scene = load_scene(args.map_dir, tiles = ["blabla"]) # ACHTUNG: tiles = ["672496d2", "672496d1"] ist hardcoded in der Funktion load_scene
    for i in range(config['experiment']['n_iterations']):
        print("########### Running iteration: ", i, "###########")


        if i < 82:
            continue

        no_errors = False
        while no_errors == False:
            try:
                scene, camera, query_pose, guessed_pose, config = initialisation(config, scene = scene)
                time_stamp = time.time()
                loc = Localizer(scene, camera, query_pose, guessed_pose, config, args, index = i)
                loc.run(time_stamp)
                no_errors = True
            except AttributeError as ae:
                print(ae)
            except ValueError as ve:
                print(ve)
            except cv2.error as e:
                print(e)


# run i < 15 again for '5_5_10_10_10_45_45_45/' mit 10 samples.