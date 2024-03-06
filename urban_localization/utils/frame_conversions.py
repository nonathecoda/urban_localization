import numpy as np
import open3d
import pyrender
from hilla.geometry.camera import Orientation, PinholeCamera, Pose
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
    if isinstance(camera, pyrender.camera.IntrinsicsCamera) or isinstance(camera, pyrender.camera.PerspectiveCamera):
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

def rotation_matrix_to_yaw_pitch_roll(R):
    # Ensure the matrix is a NumPy array
    R = np.array(R)
    
    # Calculate pitch
    pitch = np.arcsin(-R[2, 0])
    
    # Check for gimbal lock
    if np.abs(np.cos(pitch)) > 1e-6:  # Not in gimbal lock
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
    else:  # Gimbal lock
        print("Gimbal lock")
        yaw = 0  # Cannot determine yaw in gimbal lock
        roll = np.arctan2(-R[0, 1], R[1, 1])
    
    # Convert radians to degrees
    yaw_deg = np.degrees(yaw)
    pitch_deg = np.degrees(pitch)
    roll_deg = np.degrees(roll)
    ic(yaw_deg, pitch_deg, roll_deg)

    yaw_deg_updated = 360 - yaw_deg
    if yaw_deg_updated > 360:
        yaw_deg_updated = yaw_deg_updated - 360
    pitch_deg_updated = -90 + roll_deg
    if pitch_deg_updated > 360:
        pitch_deg_updated = pitch_deg_updated - 360
    roll_deg_updated = pitch_deg
    
    return yaw_deg_updated, pitch_deg_updated, roll_deg_updated

def registration_to_realworldframe(scene, camera, best_estimate, predicted_transform):
        source_extrinsic = best_estimate.camera_pose.extrinsic_matrix
        predicted_extrinsic = predicted_transform @ source_extrinsic
        predicted_pose = np.linalg.inv(predicted_extrinsic)
        hilla_pose = Pose.from_camera_in_world(rotation = Orientation.create(predicted_pose[:3, :3]), position = predicted_pose[:3, 3])
        pyrender_pose = convert_camera_pose_hilla2pyrender(hilla_pose)
        return pyrender_pose