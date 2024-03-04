from datetime import datetime
from icecream import ic
import cv2
import pyrender
from matplotlib import pyplot as plt
import numpy as np

from utils.frame_conversions import ensure_pyrender_camera, ensure_pyrender_camera_pose

def live_view(scene):
    pyrender.Viewer(scene, use_raymond_lighting=True)


def save_rendered(scene, output_dir):
    r = pyrender.OffscreenRenderer(640, 480)
    color, _ = r.render(scene)
    moment = datetime.now()
    file = output_dir / f"{moment.isoformat()}.png"
    cv2.imwrite(str(file), color[..., ::-1])


def show_rendered(scene):
    r = pyrender.OffscreenRenderer(640, 480)
    color, _ = r.render(scene)
    plt.figure()
    plt.imshow(color)
    plt.show()


def render_rgb_and_depth(scene: pyrender.Scene, camera=None, pose=None):
    if camera is not None and pose is not None:
        camera = ensure_pyrender_camera(camera)
        pose = ensure_pyrender_camera_pose(pose)
        camera_node = scene.add(camera, pose=pose)
        scene.main_camera_node = camera_node
    r = pyrender.OffscreenRenderer(640, 480)
    return r.render(scene)



    
    