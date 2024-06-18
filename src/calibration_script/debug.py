#%%
from calibration_utils import get_mis_calibrate_matrix, plt_flow_on_im, plt_depth, plt_depth_on_im, plt_im
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.nuscenes import NuScenes
import os
from os.path import join
import skimage.io as io
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyquaternion import Quaternion
from functools import reduce
import numpy as np
import torch

def downsample_im(im, downsample_scale, y_cutoff):
    h_im, w_im = im.shape[0:2]        
    h_im = int( h_im / downsample_scale )
    w_im = int( w_im / downsample_scale ) 
    
    im = resize(im, (h_im,w_im,3), order=1, preserve_range=True, anti_aliasing=False) 
    im = im.astype('uint8')
    im = im[y_cutoff:,...]
    return im

def get_2D_lidar_projection(pcl, cam_intrinsic, downsample_scale, y_cutoff):
    """投影点云到图像平面

    Args:
        pcl (_type_): 相机坐标系的点云(3, :)
        cam_intrinsic (_type_): 相机内参
        downsample_scale (_type_): 降采样比例
        y_cutoff (_type_): y轴截断值
    Returns:
        pcl_uv: 点云的像素坐标
        pcl_z: 点云在每个像素上对应的深度

    """
    pcl_xyz = cam_intrinsic @ pcl
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]
    
    # 应用 downsample_scale 和 y_cutoff
    pcl_uv = pcl_uv / downsample_scale
    pcl_uv[:, 1] = pcl_uv[:, 1] - y_cutoff
    
    return pcl_uv, pcl_z

def lidar_project_depth(pc, cam_calib, img_shape, downsample_scale=1.0, y_cutoff=0, radius=0):
    """获取点云的深度图

    Args:
        pc (_type_): 已经转到相机坐标系的点云
        cam_calib (_type_): 相机内参
        img_shape (_type_): 图像尺寸
        downsample_scale (_type_): 降采样比例
        y_cutoff (_type_): y轴截断值
    Returns:
        depth_img: 点云的深度图(1, H, W)
        pcl_uv: 点云的像素坐标(N, 2)
    """
    pc = pc[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.detach().cpu().numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc, cam_intrinsic, downsample_scale, y_cutoff)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (
            pcl_uv[:, 1] > 0) & (pcl_uv[:, 1] < img_shape[0]) & (
            pcl_z > 0)  # 筛选出图像内且深度大于0的点
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.float32)
    
    # 在投影點周圍的範圍內設置相同的深度值
    # 定義周圍像素的半徑範圍
    for i in range(pcl_uv.shape[0]):
        u, v = pcl_uv[i, 0], pcl_uv[i, 1]
        depth_value = pcl_z[i, 0]
        depth_img[max(0, v - radius):min(img_shape[0], v + radius + 1),
                max(0, u - radius):min(img_shape[1], u + radius + 1)] = depth_value
    
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.permute(2, 0, 1)
    
    return depth_img, pcl_uv

def cal_matrix_refCam_from_global(cam_data):
    ref_pose_rec = nusc.get('ego_pose', cam_data['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])    
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)    
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)        
    M_ref_from_global = reduce(np.dot, [ref_from_car, car_from_global])
    
    return M_ref_from_global

#%%
if __name__ == '__main__':
    dataset_path = '/home/yonglin/datasets/nuscenes'
    sample_idx = 0
    downsample_scale = 4
    y_cutoff = 33
    min_distance = 1
    max_t = 0.2
    max_r = 2.0
    img_shape = (192, 400)
    # img_shape = (900, 1600)
    
    nusc = NuScenes('v1.0-mini', dataroot = dataset_path, verbose=False)
    
    sample_rec = nusc.sample[sample_idx]
    cam_token = sample_rec['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_token)
    
    cam_token2 = cam_data['next']                    
    cam_data2 = nusc.get('sample_data', cam_token2)
    cam_path2 = join(nusc.dataroot, cam_data2['filename'])
    im2 = io.imread(cam_path2)
    
    cam_token3 = cam_data2['next']        
    cam_data3 = nusc.get('sample_data', cam_token3)
    cam_path3 = join(nusc.dataroot, cam_data3['filename'])
    im3 = io.imread(cam_path3)
    
    im = downsample_im(im2, downsample_scale, y_cutoff)
    im_next = downsample_im(im3, downsample_scale, y_cutoff)
    
    
    radar_token = sample_rec['data']['RADAR_FRONT']        
    radar_sample = nusc.get('sample_data', radar_token)
    
    radar_token = radar_sample['next']
    radar_sample = nusc.get('sample_data', radar_token)     # make next radar frame the latest frame
             
    pcl_path = os.path.join(nusc.dataroot, radar_sample['filename'])
    RadarPointCloud.disable_filters()
    current_pc = RadarPointCloud.from_file(pcl_path)   
    current_pc.remove_close(min_distance)
    
    
    M_refCam_from_global = cal_matrix_refCam_from_global(cam_data2)
    
    current_pose_rec = nusc.get('ego_pose', radar_sample['ego_pose_token'])
    global_from_car = transform_matrix(current_pose_rec['translation'],
                                        Quaternion(current_pose_rec['rotation']), inverse=False)

    current_cs_rec = nusc.get('calibrated_sensor', radar_sample['calibrated_sensor_token'])
    car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                        inverse=False)

    mis_calibrate_matrix, rot_error, tr_error = get_mis_calibrate_matrix(max_t, max_r)
    trans_matrix = reduce(np.dot, [M_refCam_from_global, global_from_car, car_from_current, mis_calibrate_matrix])
    
    current_pc.transform(trans_matrix)
    
    cs_rec = nusc.get('calibrated_sensor', cam_data2['calibrated_sensor_token']) 
    K = np.array(cs_rec['camera_intrinsic']) 
    
    depth_img, pcl_uv = lidar_project_depth(torch.from_numpy(current_pc.points), torch.from_numpy(K), img_shape, downsample_scale=downsample_scale, y_cutoff=y_cutoff)

    # plt_depth_on_im(depth_img.squeeze(), im2)
    plt_depth_on_im(depth_img.squeeze(), im)
# %%
    trans_matrix
# %%
