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
    def create_from_scene_view(
        cls, scene: Scene, camera: PinholeCamera, camera_pose: Pose, name: str = 'default'
    ) -> Self:
        color, depth = render_rgb_and_depth(scene, camera, camera_pose)
        target_path = '/Users/antonia/dev/masterthesis/render_hilla/v2.0/images/' + name + '.jpg'
        cv2.imwrite(target_path, color)
        return PoseEstimate(rgb = color, depth_map = depth, pc = None, camera = camera, camera_pose = camera_pose, correspondences=None, inliers = None, score = None)

    def get_score(self, inliers):
        score = 0
        for i in inliers:
            if i == True:
                score = score + 1
        return score    
