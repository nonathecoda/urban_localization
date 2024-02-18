from utils.pose_estimate import PoseEstimate
from hilla.geometry.camera import Orientation, PinholeCamera, Pose
from utils.LoFTR import inference, draw_loftr

from icecream import ic
import numpy as np

def initialise_particles(scene, camera, guessed_pose, n=100, bounds = None):
    particles = np.empty((n), dtype=object)
    for i in range(n):
        position = [np.random.uniform(low=bounds[0, 0], high=bounds[1,0]), np.random.uniform(low=bounds[0, 1], high=bounds[1,1]), 100]
        orientation = [0, -90, 0]
        weight = 1 / distance_to_guessed_pose(position, guessed_pose)
        new_estimate_pose = Pose.from_camera_in_world(
            Orientation.from_yaw_pitch_roll(np.radians(orientation)),
            position=position)
        new_estimate = PoseEstimate.create_from_scene(scene, camera, new_estimate_pose, name = str(i), particle_weight=weight, draw = False)
        particles[i] = new_estimate
    return particles

def distance_to_guessed_pose(position, guessed_pose):
    guessed_pose_position = [guessed_pose.camera_pose.position.x_coord, guessed_pose.camera_pose.position.y_coord, guessed_pose.camera_pose.position.z_coord]
    return np.linalg.norm(np.array(position) - np.array(guessed_pose_position))

def compute_weights(query_pose, particles):
    # calculate weight based on Lofter score
    weights = []
    for i, particle in enumerate(particles):
        correspondences, inliers = inference(query_pose.rgb, particle.rgb, draw = False)
        score = particle.get_score(inliers)
        particle = particle._replace(correspondences = correspondences, inliers = inliers, score = score, particle_weight = score)
        particles[i] = particle
        weights.append(score)
    return particles, weights

def resample_particles(particles, weights):
    NUM_PARTICLES = len(particles)
    probabilities = weights / np.sum(weights)
    index_numbers = np.random.choice(
        NUM_PARTICLES,
        size=NUM_PARTICLES,
        p=probabilities)
    particles = particles[index_numbers]
    ic(weights)
    ic(index_numbers)
    return particles
