import math
import open3d as o3d
import numpy as np
from icecream import ic
import copy
from utils.camera_conversions import convert_camera_model_hilla2open3d

def create_pointclouds(pose_estimate, mask, camera):   
    open3d_camera = convert_camera_model_hilla2open3d(camera)
    pose_estimate = pose_estimate._replace(pc = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(mask), open3d_camera))
    return pose_estimate

def mask_depthmaps(real_position, best_estimate):
    '''
    Mask the depthmaps so that only the points which are in the correspondence
    and only points that are not zero in either depthmap are kept
    '''
    
    skip_index = []

    # Real position
    mask_depthmap_real_position = np.zeros_like(real_position.depth_map)
    for count, p in enumerate(best_estimate.correspondences['keypoints0'].numpy()):
        
        x = math.floor(p[0])
        y = math.floor(p[1])
        
        if real_position.depth_map[y][x] != 0.:
            mask_depthmap_real_position[y][x] = real_position.depth_map[y][x]
        else:
            skip_index.append(count)

    # Estimated position
    mask_depthmap_estimate = np.zeros_like(best_estimate.depth_map)
    for count, p in enumerate(best_estimate.correspondences['keypoints1'].numpy()):
        x = math.floor(p[0])
        y = math.floor(p[1])
        
        if best_estimate.depth_map[y][x] != 0.:
            mask_depthmap_estimate[y][x] = best_estimate.depth_map[y][x]
        else:
            skip_index.append(count)

    # Make sure none of the pcd has a point which is zero for the other pointcloud:
    for count, p in enumerate(best_estimate.correspondences['keypoints0'].numpy()):
        if count in skip_index:

            x_real = math.floor(best_estimate.correspondences['keypoints0'].numpy()[count][0])
            y_real = math.floor(best_estimate.correspondences['keypoints0'].numpy()[count][1])
            
            x_est = math.floor(best_estimate.correspondences['keypoints1'].numpy()[count][0])
            y_est = math.floor(best_estimate.correspondences['keypoints1'].numpy()[count][1])


            mask_depthmap_estimate[y_est][x_est] = 0.
            mask_depthmap_real_position[y_real][x_real] = 0.

    return mask_depthmap_real_position, mask_depthmap_estimate

def get_centroid_translation(pcd_0, pcd_1):
    '''
    Calculate the centroid of two pointclouds and return the transformation matrix to align the two pointclouds
    '''
    xyz_0 = np.asarray(pcd_0.points)
    xyz_1 = np.asarray(pcd_1.points)

    #Calculate centroid 0  
    x0 = np.mean([x[0] for x in xyz_0])
    y0 = np.mean([x[1] for x in xyz_0])
    z0 = np.mean([x[2] for x in xyz_0])

    #Calculate centroid 1
    x1 = np.mean([x[0] for x in xyz_1])
    y1 = np.mean([x[1] for x in xyz_1])
    z1 = np.mean([x[2] for x in xyz_1])
    
    transformation_matrix = np.array(
            [[1.,   0.,     0.,     (x0-x1)],
            [ 0.,   1.,     0.,     (y0-y1)],
            [ 0.,   0.,     1.,     (z0-z1)],
            [ 0.,   0.,     0.,     1.]])

    return transformation_matrix

def run_icp(source, target):
    '''
    Run ICP registration on two pointclouds
    '''
    trans_init = get_centroid_translation(target.pc, source.pc)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source.pc,
        target.pc,
        3,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    return reg_p2p

def draw_registration_result(source, target, transformation, transform = True):
    source_temp = copy.deepcopy(source.pc)
    target_temp = copy.deepcopy(target.pc)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    #source_temp.transform(transformation)
    if transform == False:
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
        return
    source_temp.transform(transformation)
    if transform ==  True:
        ic("transform")
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])