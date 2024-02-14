from icecream import ic
import argparse
import numpy as np

from utils.load_scene import load_scene
from utils.pose_estimate import PoseEstimate
from hilla.geometry.camera import Orientation, PinholeCamera, Pose

class Localizer():
    def __init__(self):
        self.__location = "Japan"

    def run(self):
        ic(self.__location)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Script to export the keypoints for images in a dataset using a base detector')
    #parser.add_argument('-y', '--yaml-config', default='configs/config_export_keypoints.yaml', help='YAML config file')
    parser.add_argument('-i', '--image-dir', default = '/Users/antonia/dev/masterthesis/urban_localization/images/', help='Image directors')
    parser.add_argument('-o', '--obj-dir', default='/Users/antonia/dev/masterthesis/Helsinki3D_2017_OBJ_672496x2/672496a1', help='Directory of the obj files for the 3d model')

    args = parser.parse_args()

    

    def initialisation():
        print("Initialising stuff")
        # create pyrender scene object
        scene = load_scene(args.obj_dir)
        # create hilla camera object
        camera = PinholeCamera.from_fov(width=640, height=480, hfov_deg=90)
        # create PoseEstimate object for true pose
        true_position = [scene.centroid[0], scene.centroid[1] + 50, 60]
        true_orientation = [-90, -40, 0]
        true_pose_hilla = Pose.from_camera_in_world(
            Orientation.from_yaw_pitch_roll(np.radians(true_position)),
            position=true_orientation,
        )
        true_pose = PoseEstimate.create_from_scene_view(scene, camera, true_pose_hilla, name = 'true_pose')
        # create PoseEstimate object for guessed pose
        guessed_position = np.add(true_position, [50, -45, 30])
        guessed_orientation = np.add(true_orientation, [10, 0, 0])
        guessed_pose_hilla = Pose.from_camera_in_world(
            Orientation.from_yaw_pitch_roll(np.radians(guessed_position)),
            position=guessed_orientation,
        )
        guessed_pose = PoseEstimate.create_from_scene_view(scene, camera, guessed_pose_hilla, name = 'guessed_pose')
        return scene, camera, true_pose, guessed_pose

    
    loc = Localizer()
    scene, camera, true_pose, guessed_pose = initialisation()
    loc.run(scene, camera, true_pose, guessed_pose)