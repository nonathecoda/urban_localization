import numpy as np
import pyrender
import open3d as o3d
import trimesh
from hilla.geometry.camera import PinholeCamera
from pyrender import Scene
from pathlib import Path
import os
import random
from icecream import ic
import tqdm
import math

from utils.load_scene import load_scene, choose_random_map, change_poses, load_entire_scene
from utils.pose_estimate import PoseEstimate
from utils.frame_conversions import convert_camera_model_hilla2pyrender
from search_strategy.heuristic_sampling import get_best_pose_estimate, sample_estimates_gauss
from utils.pointcloud_tools import create_pointclouds, mask_depthmaps, run_icp, draw_registration_result
from utils.frame_conversions import convert_camera_pose_hilla2pyrender, rotation_matrix_to_yaw_pitch_roll, registration_to_realworldframe
from hilla.geometry.camera import Orientation, PinholeCamera, Pose
from utils.LoFTR import draw_loftr
import yaml


def load_obj(data_dir, scaling_factor = 0, translation = [0,0,0], rotation = [0,0,0]) -> pyrender.Mesh:    
    tm_mesh = trimesh.load(data_dir + "/textured.obj")
    
    # Create a scaling matrix for scaling the mesh by a factor of 100
    scaling_matrix = np.eye(4)  # Create an identity matrix
    scaling_matrix[0, 0] = scaling_factor  # Scale x
    scaling_matrix[1, 1] = scaling_factor  # Scale y
    scaling_matrix[2, 2] = scaling_factor  # Scale z
    tm_mesh.apply_transform(scaling_matrix)

    # Create a translation matrix to move the mesh
    translation_x = translation[0]
    translation_y = translation[1]
    translation_z = translation[2] 
    translation_matrix = np.eye(4)  # Identity matrix for translation
    translation_matrix[0, 3] = translation_x  # Move along x
    translation_matrix[1, 3] = translation_y  # Move along y
    translation_matrix[2, 3] = translation_z  # Move along z
    tm_mesh.apply_transform(translation_matrix)
    

    # Create a rotation matrix to rotate the mesh
    # Rotation around the Y-axis

    # first,move object to origin
    centroid = tm_mesh.centroid
    tm_mesh.apply_translation(-tm_mesh.centroid)

    # rotate object
    alpha_deg, beta_deg, gamma_deg = rotation[0], rotation[1], rotation[2]
    alpha, beta, gamma = np.radians(alpha_deg), np.radians(beta_deg), np.radians(gamma_deg)
    origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    Rx = trimesh.transformations.rotation_matrix(alpha, xaxis)
    Ry = trimesh.transformations.rotation_matrix(beta, yaxis)
    Rz = trimesh.transformations.rotation_matrix(gamma, zaxis)
    R = trimesh.transformations.concatenate_matrices(Rx, Ry, Rz)
    tm_mesh.apply_transform(R)

    # move object back
    tm_mesh.apply_translation(centroid)
    mesh = pyrender.Mesh.from_trimesh(tm_mesh)
    ic(mesh.centroid)
    return mesh

def _load_scene(data_dir, camera_pose = None) -> Scene:
    scene = pyrender.Scene(ambient_light=np.ones(3))
    for folder in tqdm.tqdm(os.listdir(data_dir)):
        
        if (folder != "672496d2") and (folder != "672496d1"):# and (folder != "672496d4") and (folder != "672496d3"):
            continue
        tiles = Path(data_dir).rglob(folder + "/*L21*.obj")
        for tile_path in tiles:
            tm_mesh = trimesh.load(tile_path)
            mesh = pyrender.Mesh.from_trimesh(tm_mesh)
            scene.add(mesh)
    # add camera
    if camera_pose is not None:
        camera = pyrender.camera.PerspectiveCamera(yfov = np.pi/2)
        scene.add(camera, pose=camera_pose)
    scene.ambient_light = np.ones(3)
    #pyrender.Viewer(scene, use_raymond_lighting=True)
    ic(scene.bounds)
    return scene


#############################################################################
############################MAIN#############################################

with open('/Users/antonia/dev/masterthesis/urban_localization/urban_localization/configs/config_real_data.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

obj_dir  = "/Users/antonia/Downloads/obj_1/textured.obj"
map_dir = "/Users/antonia/dev/masterthesis/Helsinki3D_2017_OBJ_672496x2/"

## load scene
scene = _load_scene(map_dir)
ic(scene.centroid)
## load and scale obj
#translation = [6625., 4750., 40]
obj = load_obj(obj_dir, scaling_factor=110, translation = [6625.+50, 4750-120., 35], rotation = [90, -135, 0])
ic(obj.centroid)
### create camera
hilla_camera = PinholeCamera.from_fov(width=640, height=480, hfov_deg=90)
pyrender_camera = convert_camera_model_hilla2pyrender(hilla_camera)

#### create query pose
query_pose_hilla = Pose.from_camera_in_world(
    Orientation.from_yaw_pitch_roll(np.radians(config['poses']['query']['orientation'])),
    position=config['poses']['query']['position'],
)
pyrender_pose = convert_camera_pose_hilla2pyrender(query_pose_hilla)

## scale obj and add it to scene
scene.add(obj)
scene.add(pyrender_camera, pose=pyrender_pose)
pyrender.Viewer(scene, use_raymond_lighting=True)

exit()

pyrender_pose = convert_camera_pose_hilla2pyrender(query_pose_hilla)
scene.add(pyrender_camera, pose=pyrender_pose)

ic(scene.centroid)
pyrender.Viewer(scene, use_raymond_lighting=True)
#############################################################################
'''
cube = trimesh.primitives.Box(extents=(1.0, 1.0, 1.0))

# Define the color red in RGBA format (Red, Green, Blue, Alpha)
red_color = [1.0, 0.0, 0.0, 1.0]  # Red with full opacity

# Create a material for the cube with the specified color
material = pyrender.MetallicRoughnessMaterial(baseColorFactor=red_color)

# Create a mesh from the trimesh object with the defined material
cube_mesh = pyrender.Mesh.from_trimesh(cube, material=material)

# Step 3: Add the Cube Mesh to the Scene
scene.add(cube_mesh, pose=pyrender_pose)
'''

#############################################################################