import numpy as np
import pyrender
import trimesh
from hilla.geometry.camera import PinholeCamera
from pyrender import Scene
from pathlib import Path

from utils.camera_conversions import convert_camera_model_hilla2pyrender

def load_scene(data_dir, camera_pose = None) -> Scene:
    scene = pyrender.Scene(ambient_light=np.ones(3))
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
    return scene

def create_camera() -> pyrender.camera.IntrinsicsCamera:
    hilla_camera = PinholeCamera.from_fov(width=640, height=480, hfov_deg=90)
    return convert_camera_model_hilla2pyrender(hilla_camera)