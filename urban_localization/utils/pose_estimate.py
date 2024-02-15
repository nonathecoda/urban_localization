import copy
from typing import NamedTuple

import numpy as np
import open3d as o3d
from hilla.geometry.camera import Orientation, PinholeCamera, Pose
from icecream import ic
from numpy.linalg import inv
#from open3d.geometry import Image, PointCloud, RGBDImage
from pyrender import Scene
from typing_extensions import Self
import cv2

from utils.camera_conversions import convert_camera_model_hilla2open3d
from utils.render import render_rgb_and_depth


class PoseEstimate(NamedTuple):
    rgb: np.ndarray
    depth_map: np.ndarray
    pc: o3d.geometry.PointCloud
    camera: PinholeCamera
    camera_pose: Pose
    correspondences: np.array 
    inliers: np.array
    score: int

    @classmethod
    def create_from_scene(
        cls, scene: Scene, camera: PinholeCamera, camera_pose: Pose, name: str = 'default'
    ) -> Self:
        color, depth = render_rgb_and_depth(scene, camera, camera_pose)
        target_path = '/Users/antonia/dev/masterthesis/render_hilla/v2.0/images/' + name + '.jpg'
        cv2.imwrite(target_path, color)
        return PoseEstimate(rgb = color, depth_map = depth, pc = None, camera = camera, camera_pose = camera_pose, correspondences=None, inliers = None, score = None)

    @classmethod
    def create_from_image(cls, camera: PinholeCamera, camera_pose: Pose, name: str = 'default', config = None, args = None) -> Self:
        
        
        # load images
        stereo = cv2.imread(args.query_image)
        frame_left = stereo[0:config['camera']['height'], 0:config['camera']['width']]
        frame_right = stereo[0:config['camera']['height'], config['camera']['width']:2*config['camera']['width']]

        # undistort images
        intrinsics_left = np.load(config['camera']['intrinsics_left'])
        distortion_left = np.load(config['camera']['distortion_left'])
        
        intrinsics_right = np.load(config['camera']['intrinsics_right'])
        distortion_right = np.load(config['camera']['distortion_right'])
        
        frame_left = cv2.undistort(frame_left, intrinsics_left, distortion_left)
        frame_right = cv2.undistort(frame_right, intrinsics_right, distortion_right)

        #rectify images
        stereomap_file = cv2.FileStorage()
        stereomap_file.open(config['camera']['stereo']['path_stereomap'], cv2.FileStorage_READ)

        stereoMapL_x = stereomap_file.getNode('stereoMapL_x').mat()
        stereoMapL_y = stereomap_file.getNode('stereoMapL_y').mat()
        stereoMapR_x = stereomap_file.getNode('stereoMapR_x').mat()
        stereoMapR_y = stereomap_file.getNode('stereoMapR_y').mat()

        frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        #compute disparity
        stereo = cv2.StereoBM.create(numDisparities=96, blockSize=5)
        stereo.setTextureThreshold(10)
        disparity = stereo.compute(frame_left,frame_right)

        #convert disparity to depth
        depth_map = (intrinsics_left[0,0]* config['camera']['stereo']['baseline']) / disparity # TODO: why intrinsics_left[0,0] and not intrinsics_right[0,0]?

        return PoseEstimate(rgb = frame_left, depth_map = depth_map, pc = None, camera = camera, camera_pose = camera_pose, correspondences=None, inliers = None, score = None)

    def get_score(self, inliers):
        score = 0
        for i in inliers:
            if i == True:
                score = score + 1
        return score    
