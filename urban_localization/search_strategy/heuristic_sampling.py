from icecream import ic
from hilla.geometry.camera import Orientation, PinholeCamera, Pose
from operator import attrgetter
from utils.LoFTR import inference, draw_loftr
import numpy as np
from utils.pose_estimate import PoseEstimate

def get_best_pose_estimate(true_pose, sampled_pose_estimates):
    estimates_copy = []
    for sample in sampled_pose_estimates:
        correspondences, inliers = inference(true_pose.rgb, sample.rgb)
        score = sample.get_score(inliers)
        sample = sample._replace(correspondences = correspondences, inliers = inliers, score = score)
        estimates_copy.append(sample)
        
    best_estimate = max(estimates_copy, key=attrgetter('score'))
    draw_loftr(true_pose, best_estimate)
    return best_estimate

def sample_estimates(scene, camera, guessed_pose, n):
    '''
    in transforms, each line represents by how many metres / degrees the respective value should be shifted
    [x,y,z],[yaw,pitch,roll]
    '''
    guessed_pose_position = [guessed_pose.camera_pose.position.x_coord, guessed_pose.camera_pose.position.y_coord, guessed_pose.camera_pose.position.z_coord]
    guessed_pose_orientation = np.rad2deg(guessed_pose.camera_pose.orientation.yaw_pitch_roll)

    estimates = []
    transforms = [[[ -20,  -20,   0],[0,  0,  0]],
                  [[ -10,  -10,   0],[0,  0,  0]],
                  [[  10,  -10,   0],[0,  0,  0]],
                  [[ -10,   10,   0],[0,  0,  0]],
                  [[  10,   10,  40],[0,  0,  0]],
                  [[ -10,  -10,  40],[0,  0,  0]],
                  [[  10,  -10,  40],[0,  0,  0]],
                  [[ -10,   10,  40],[0,  0,  0]],
                  [[  10,   10, -40],[0,  0,  0]],
                  [[ -10,  -10, -40],[0,  0,  0]],
                  [[  10,  -10, -40],[0,  0,  0]],
                  [[ -10,   10, -40],[0,  0,  0]]]
                
    for count, transform in enumerate(transforms):
        if count == 1:
            break
        new_estimate_pose = Pose.from_camera_in_world(
            Orientation.from_yaw_pitch_roll(np.radians(np.add(guessed_pose_orientation, transform[1]))),
            position=np.add(guessed_pose_position, transform[0])
        )
        new_estimate = PoseEstimate.create_from_scene_view(scene, camera, new_estimate_pose, name = str(count))
                    
        estimates.append(new_estimate)

    return estimates