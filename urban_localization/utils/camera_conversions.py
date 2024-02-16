import numpy as np
import open3d
import pyrender
from hilla.geometry.camera import PinholeCamera, Pose
#from utils.pose_estimate import PoseEstimate
from icecream import ic

def convert_camera_pose_hilla2pyrender(hilla_pose: Pose) -> np.ndarray:
    """
    For Pyrender conventions, see:
    https://pyrender.readthedocs.io/en/latest/examples/cameras.html

    TLDR:
    Hilla's default camera (image):
        X increases to the right
        Y increases downwards
        Z is into the screen (along your sight)

    Pyrender's default camera (image):
        X increases to the right
        Y increases upwards
        Z is coming at you

    I.e. we should flip both, Y and Z, but leave translation be.
    """
    position = hilla_pose.position.coords

    flip_yz = np.diag([1, -1, -1])
    hilla_orientation = hilla_pose.orientation.camera_to_world_matrix
    pyrender_orientation = hilla_orientation @ flip_yz

    pyrender_pose = np.eye(4)
    pyrender_pose[:3, :3] = pyrender_orientation
    pyrender_pose[:3, 3] = position

    return pyrender_pose


def ensure_pyrender_camera_pose(pose) -> np.ndarray:
    if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
        return pose
    if isinstance(pose, Pose):
        return convert_camera_pose_hilla2pyrender(pose)
    raise TypeError


def convert_camera_model_hilla2pyrender(
    hicam: PinholeCamera,
) -> pyrender.camera.IntrinsicsCamera:
    return pyrender.camera.IntrinsicsCamera(
        fx=hicam.fx,
        fy=hicam.fy,
        cx=hicam.px,
        cy=hicam.py,
        znear=0.01,
        zfar=1000.0,
    )


def ensure_pyrender_camera(camera) -> pyrender.camera.IntrinsicsCamera:
    if isinstance(camera, pyrender.camera.IntrinsicsCamera):
        return camera
    if isinstance(camera, PinholeCamera):
        return convert_camera_model_hilla2pyrender(camera)
    raise TypeError


def convert_camera_model_hilla2open3d(
    hicam: PinholeCamera,
) -> open3d.camera.PinholeCameraIntrinsic:
    width = hicam.resolution.width
    height = hicam.resolution.height
    fx = hicam.intrinsic_matrix[0, 0]
    fy = hicam.intrinsic_matrix[1, 1]
    cx = hicam.intrinsic_matrix[0, 2]
    cy = hicam.intrinsic_matrix[1, 2]
    #width: int, height: int, fx: float, fy: float, cx: float, cy: float)
    #return open3d.camera.PinholeCameraIntrinsic(
    #    width=hicam.resolution.width,
    #    height=hicam.resolution.height,
    #    intrinsic_matrix=hicam.intrinsic_matrix,
    #)
    return open3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

