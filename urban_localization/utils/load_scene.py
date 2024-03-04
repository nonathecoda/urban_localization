import numpy as np
import pyrender
import trimesh
from hilla.geometry.camera import PinholeCamera
from pyrender import Scene
from pathlib import Path
import os
import random
from icecream import ic
import tqdm
from utils.frame_conversions import convert_camera_model_hilla2pyrender

def load_scene(data_dir, camera_pose = None, tiles = None) -> Scene:
    scene = pyrender.Scene(ambient_light=np.ones(3))
    
    if tiles is not None:
        for folder in tqdm.tqdm(os.listdir(data_dir)):
            if (folder != "672496d2") and (folder != "672496d1"):# and (folder != "672496d4") and (folder != "672496d3"):
                continue
            tiles = Path(data_dir).rglob(folder + "/*L21*.obj")
            for tile_path in tiles:
                tm_mesh = trimesh.load(tile_path)
                mesh = pyrender.Mesh.from_trimesh(tm_mesh)
                scene.add(mesh)
    else:
        tiles = Path(data_dir).rglob("*L21*.obj")
        for tile_path in tiles:
            tm_mesh = trimesh.load(tile_path)
            mesh = pyrender.Mesh.from_trimesh(tm_mesh)
            scene.add(mesh)
    scene.ambient_light = np.ones(3)
    if camera_pose is not None:
        camera = create_camera()
        scene.add(camera, pose=camera_pose)
    #pyrender.Viewer(scene, use_raymond_lighting=True)
    ic(scene.bounds)
    return scene

def load_entire_scene(data_dir, camera_pose = None) -> Scene:
    folder_dir = os.path.dirname(data_dir)
    obj_folders = [name for name in os.listdir(folder_dir) if os.path.isdir(os.path.join(folder_dir, name))]
    scene = pyrender.Scene(ambient_light=np.ones(3))
    for folder in tqdm.tqdm(obj_folders):
        tiles = Path(os.path.join(folder_dir, folder)).rglob("*L21*.obj")
        for tile_path in tiles:
            tm_mesh = trimesh.load(tile_path)
            mesh = pyrender.Mesh.from_trimesh(tm_mesh)
            scene.add(mesh)
    scene.ambient_light = np.ones(3)
    if camera_pose is not None:
        camera = create_camera()
        scene.add(camera, pose=camera_pose)
    #pyrender.Viewer(scene, use_raymond_lighting=True)
    ic(scene.bounds)
    return scene

def create_camera() -> pyrender.camera.IntrinsicsCamera:
    hilla_camera = PinholeCamera.from_fov(width=640, height=480, hfov_deg=90)
    return convert_camera_model_hilla2pyrender(hilla_camera)

def choose_random_map(data_dir) -> str:
    folder_dir = os.path.dirname(data_dir)
    obj_folders = [name for name in os.listdir(folder_dir) if os.path.isdir(os.path.join(folder_dir, name))]
    # choose random map part by looping through the directory
    random_folder = random.choice(obj_folders)
    random_data_dir = str(os.path.join(folder_dir, random_folder))
    print("random_data_dir: ", random_data_dir)
    return random_data_dir

def change_poses(config, scene) -> dict:
    bounds_x = (scene.bounds[0,0], scene.bounds[1,0])
    bounds_y = (scene.bounds[0,1], scene.bounds[1,1])
    
    config['poses']['query']['position'] = [random.uniform(bounds_x[0], bounds_x[1]), random.uniform(bounds_y[0], bounds_y[1]), random.uniform(config['experiment']['min_altitude'], config['experiment']['max_altitude'])]
    config['poses']['query']['orientation'] = [random.uniform(0, 360), np.random.normal(config['experiment']['alpha'], config['sampling']['sd_query_pitch']),np.random.normal(0,config['sampling']['sd_query_roll'])]

    noise_position_x = np.random.normal(config['poses']['query']['position'][0], config['sampling']['sd_guess_x'])
    noise_position_y = np.random.normal(config['poses']['query']['position'][1], config['sampling']['sd_guess_y'])
    noise_position_z = np.random.normal(config['poses']['query']['position'][2], config['sampling']['sd_guess_z'])
    config['poses']['guess']['position'] = [noise_position_x, noise_position_y, noise_position_z]

    noise_orientation_yaw = np.random.normal(config['poses']['query']['orientation'][0], config['sampling']['sd_guess_yaw'])
    noise_orientation_pitch = np.random.normal(config['poses']['query']['orientation'][1], config['sampling']['sd_guess_pitch'])
    noise_orientation_roll = np.random.normal(config['poses']['query']['orientation'][2], config['sampling']['sd_guess_roll'])
    config['poses']['guess']['orientation'] = [noise_orientation_yaw, noise_orientation_pitch, noise_orientation_roll]
    return config

def transform_mesh(tm_mesh, scaling_factor, translation, rotation):
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
    return tm_mesh

def load_obj(data_dir, pose = None, scaling_factor = 0, translation = [0,0,0], rotation = [0,0,0]) -> Scene:
    scene = pyrender.Scene(ambient_light=np.ones(3))
    tm_mesh = trimesh.load(data_dir)
    tm_mesh = transform_mesh(tm_mesh, scaling_factor, translation, rotation)
    mesh = pyrender.Mesh.from_trimesh(tm_mesh)
    scene.add(mesh)
    scene.ambient_light = np.ones(3)
    if pose is not None:
        camera = create_camera()
        scene.add(camera, pose=pose)
    #pyrender.Viewer(scene, use_raymond_lighting=True)
    ic(scene.bounds)
    return scene