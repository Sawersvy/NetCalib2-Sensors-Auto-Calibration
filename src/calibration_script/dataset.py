#%%
import torch
from torch.utils import data
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import h5py
import os
# os.environ['CUDA_VISIBLE_DEVICES']='5'acm107gpu
import cv2
from typing import Tuple, Union, Optional
from torch import Tensor

import torch
import skimage.io as io
from torchvision import transforms
import torchvision.transforms.functional as TTF
from nuscenes.nuscenes import NuScenes
import sys
from torchvision.transforms import Compose

from calibration_script.utils import inverse_normalize
# from utility.utils import inverse_normalize
from calibration_script.fuse_radar import merge_selected_radar
from calibration_script.calibration_utils import get_mis_calibrate_matrix, plt_depth, plt_depth_on_im, plt_im
import torch.backends.cudnn as cudnn

def get_2D_lidar_projection(pcl, cam_intrinsic, downsample_scale, y_cutoff):
    """Projecting a point cloud onto an image plane

    Args:
        pcl (_type_): Point cloud in camera coordinate system(3, :)
        cam_intrinsic (_type_): Camera intrinsic
        downsample_scale (_type_): downsample scale
        y_cutoff (_type_): y cutoff
    Returns:
        pcl_uv: Pixel coordinates of the point cloud
        pcl_z: Corresponding depth of the point cloud at each pixel

    """
    pcl_xyz = cam_intrinsic @ pcl
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]
    
    # apply downsample_scale and y_cutoff
    pcl_uv = pcl_uv / downsample_scale
    pcl_uv[:, 1] = pcl_uv[:, 1] - y_cutoff
    
    return pcl_uv, pcl_z

def lidar_project_depth(pc, cam_calib, img_shape, downsample_scale=1.0, y_cutoff=0):
    """Get a depth map of the point cloud

    Args:
        pc (_type_): Point cloud that has been transferred to the camera coordinate system
        cam_calib (_type_): Camera intrinsic
        img_shape (_type_): Image shape
        downsample_scale (_type_): downsample scale
        y_cutoff (_type_): y cutoff
    Returns:
        depth_img: Depth map of the point cloud(1, H, W)
        pcl_uv: Pixel coordinates of the point cloud(N, 2)
    """
    pc = pc[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib.detach().cpu().numpy()
    pcl_uv, pcl_z = get_2D_lidar_projection(pc, cam_intrinsic, downsample_scale, y_cutoff)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (
            pcl_uv[:, 1] > 0) & (pcl_uv[:, 1] < img_shape[0]) & (
            pcl_z > 0)  # Filter out points within the image and with a depth greater than 0
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1), dtype=np.float32)
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img.permute(2, 0, 1)
    
    return depth_img, pcl_uv


def load_weights(path, model):
    f_checkpoint = join(path)        
    if os.path.isfile(f_checkpoint):
        print('load best model')        
        model.load_state_dict(torch.load(f_checkpoint)['state_dict_best'])
    else:
        sys.exit('No model found')
        
    
def init_env():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:5" if use_cuda else "cpu")    
    cudnn.benchmark = True if use_cuda else False
    return device


def init_data_loader(args, split, mis_calib = False):
    
    if split == 'train':
        batch_size = args.batch_size
        if args.no_data_shuffle:
            shuffle = False
        else:
            shuffle = True
    else:
        batch_size = args.test_batch_size
        shuffle = False
        
    args_dataset = {'prepared_path': '/mnt/HDD7/yonglin/data/prepared_nuscenes_trainval',
                    'dataset_path': '/mnt/HDD7/yonglin/data/nuscenes',
                    'split': 'train',
                    'max_t': 1.0,
                    'max_r': 0.1,
                    'nuscenes_version': "v1.0-trainval"
                    }
    args_data_loader = {'batch_size': batch_size,
                       'shuffle': shuffle,
                       'num_workers': args.num_workers}
    dataset = DatasetNuscenesCalibNet(**args_dataset)    
    data_loader = torch.utils.data.DataLoader(dataset, **args_data_loader)
    
    return data_loader
    

class DatasetNuscenesCalibNet(data.Dataset):     
    def __init__(self, prepared_path, dataset_path, split, max_t, max_r, max_depth, downsample_scale, y_cutoff, nuscenes_version="v1.0-trainval", modality='radar'):               
        'Initialization'   
        
        self.rotation_offset = np.deg2rad(max_r)
        self.translation_offset = max_t
        self.rot_mean = 0
        self.trans_mean = 0
        self.rot_std = 2*self.rotation_offset/np.sqrt(12)
        self.trans_std = 2*self.translation_offset/np.sqrt(12)
        
        self.split = split
        self.nuscenes_version = nuscenes_version
        self.data = h5py.File(prepared_path, 'r')[split] 
        self.dataset_path = dataset_path
        self.max_t = max_t
        self.max_r = max_r
        self.max_depth = max_depth
        self.downsample_scale = downsample_scale
        self.y_cutoff = y_cutoff
        self.modality = modality
        self.image_shape = (192, 400)
        print(f"{split}: Load NuScenes.")
        self.nusc = NuScenes(self.nuscenes_version, dataroot = self.dataset_path, verbose=False)
        print(f"{split}: Load NuScenes Done.")
        
        print(f"{split}: Load prepared data.")
        self.im_list = self.data['im'][...]
        self.K_list = self.data['K']
        self.gt = self.data['gt'][...,[0]].astype('f4')
        self.indices = self.data['indices']
        print(f"{split}: Load Prepared Done.")
        if split == 'test':
            self.msk_lh_list = self.data['msk_lh']
        
        print(f"{split}: Load All Data Done !")
                           
    def custom_transform(self, rgb, img_rotation, h_mirror, flip = False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def get_image_shape(self):
        return self.image_shape                       
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.indices)
    
    
    def cal_radar(self, sample_idx, K):
        frm_range = [0,0]
        max_t = self.max_t
        max_r = self.max_r
        mis_calibrate_matrix, rot_error, tr_error, gt_error = get_mis_calibrate_matrix(max_t, max_r)
        if self.modality == 'radar':
            mis_x1, mis_y1, mis_depth1, mis_all_times1, mis_x2, mis_y2, mis_depth2, mis_all_times2, mis_rcs, mis_v_comp, all_mis_pc1, all_origin_pc1, all_raw_pc1, trans_matrix1= merge_selected_radar(self.nusc, sample_idx, self.modality, frm_range, mis_calibrate_matrix)

        else:
            mis_x1, mis_y1, mis_depth1, mis_all_times1, mis_x2, mis_y2, mis_depth2, mis_all_times2, all_mis_pc1, all_origin_pc1, all_raw_pc1, trans_matrix1= merge_selected_radar(self.nusc, sample_idx, self.modality, frm_range, mis_calibrate_matrix)

        return mis_calibrate_matrix, all_mis_pc1, all_origin_pc1, all_raw_pc1, trans_matrix1, rot_error, tr_error, gt_error
        
    def destandardize(self, rot: Union[Tensor, np.ndarray], trans: Union[Tensor, np.ndarray]) -> \
            Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor]]:
        return rot*self.rot_std+self.rot_mean, trans*self.trans_std+self.trans_mean

    def test_destandardize(self, rot: Union[Tensor, np.ndarray], trans: Union[Tensor, np.ndarray],
                      rot_offset: Optional[float] = None, trans_offset: Optional[float] = None) -> \
            Union[Tuple[np.ndarray, np.ndarray], Tuple[Tensor, Tensor]]:
        if rot_offset is None:
            return rot*self.rot_std+self.rot_mean, trans*self.trans_std+self.trans_mean
        else:
            rot_offset = np.deg2rad(rot_offset)
            trans_offset = trans_offset
            rot_mean = 0
            trans_mean = 0
            rot_std = 2 * rot_offset / np.sqrt(12)
            trans_std = 2 * trans_offset / np.sqrt(12)
            return rot*rot_std+rot_mean, trans*trans_std+trans_mean

    def __getitem__(self, idx):
        'Generate one sample of data'
        frame = self.indices[idx]        
        im1 = torch.from_numpy(self.im_list[idx].astype('float32').transpose((2,0,1))/255)  # (3,h,w)
        rgb_depth = np.load(os.path.join(f'/mnt/SSD1/yonglin/datasets/prepared_nuscenes_trainval/depth_anything_output/{self.split}', '%05d_depth_anything_output.npy' % frame)).astype('f4')
        rgb_depth = np.expand_dims(rgb_depth, axis=0)
        rgb_depth = torch.from_numpy(rgb_depth)
        rgb_depth = rgb_depth / (1 / self.max_depth)  # need to divide the scale from fixed max value of true depth
        
        
        K = torch.from_numpy(self.K_list[idx])
        mis_calibrate_matrix, all_mis_pc1, all_origin_pc1, all_raw_pc1, trans_matrix1, rot_error, tr_error, gt_err = self.cal_radar(frame, K)    
        gt_err = np.array(gt_err, dtype=np.float32)
        gt_err_norm = gt_err.copy()
        gt_err_norm[:3] = (gt_err_norm[:3]-self.rot_mean)/self.rot_std
        gt_err_norm[3:] = (gt_err_norm[3:]-self.trans_mean)/self.trans_std
       
        # point cloud data
        pc_raw = all_raw_pc1.points[0:3]  # The point cloud of the camera coordinate system is the target.
        pc_target = all_origin_pc1.points[0:3]  # The point cloud of the camera coordinate system is the target.
        pc_source = all_mis_pc1.points[0:3] # The perturbed point cloud is source
        
        # the output is 4xN and the last row is 1
        homogeneous = np.expand_dims(np.ones(pc_target.shape[1]), axis=0)
        pc_raw = torch.from_numpy(np.concatenate((pc_raw, homogeneous), 0)).float()
        pc_target = torch.from_numpy(np.concatenate((pc_target, homogeneous), 0)).float()
        pc_source = torch.from_numpy(np.concatenate((pc_source, homogeneous), 0)).float()
        
        mis_radar_depth_map, uv = lidar_project_depth(pc_source, K, self.image_shape, downsample_scale=self.downsample_scale, y_cutoff=self.y_cutoff)
        mis_radar_depth_map[mis_radar_depth_map>self.max_depth] = 0  
        mis_radar_depth_map = mis_radar_depth_map/self.max_depth               # normalized to 0-1
        
        # Inverse normalize, in order to match DepthAnything
        mis_radar_depth_map = inverse_normalize(mis_radar_depth_map, self.max_depth)
        
        
        # lidar_depth_map = torch.from_numpy(self.gt[idx].astype('float32').transpose((2,0,1)))
        # lidar_depth_map[lidar_depth_map>self.max_depth] = 0
        # # pose data (perturbed inverse as point cloud position)
        # i_pose_target = np.array(mis_calibrate_matrix, dtype=np.float32)
        # pose_target = i_pose_target.copy()
        # pose_target[:3, :3] = pose_target[:3, :3].T
        # pose_target[:3, 3] = -np.matmul(pose_target[:3, :3], pose_target[:3, 3])
        # pose_target = torch.from_numpy(pose_target)
        pose_source = torch.eye(4)
        trans_matrix1 = torch.from_numpy(trans_matrix1).float()
        K = K.float()
        # P = K.mm(trans_matrix1[:3, :])
        
        if self.split == 'test':
            return rgb_depth, mis_radar_depth_map, gt_err, gt_err_norm, im1, pc_raw, trans_matrix1, K
        return rgb_depth, mis_radar_depth_map, gt_err, gt_err_norm, pc_raw, trans_matrix1

        # data output in unified dictionary format
        if self.split == 'test':
            return rgb_depth, mis_radar_depth_map, gt_err, gt_err_norm, pc_raw, trans_matrix1, K
        else:
            return rgb_depth, mis_radar_depth_map, gt_err, gt_err_norm


if __name__=='__main__':
    
    args_train_set = {'prepared_path': '/mnt/SSD1/yonglin/datasets/prepared_nuscenes_mini/prepared_data.h5',
                    'dataset_path': '/mnt/SSD1/yonglin/datasets/nuscenes',
                    'split': 'test',
                    'max_t': 0,
                    'max_r': 0,
                    'max_depth': 200,
                    'downsample_scale': 4,
                    'y_cutoff': 33,
                    'nuscenes_version': "v1.0-mini",
                    'modality': 'lidar'
                    }
    args_train_loader = {'batch_size': 1,
                         'shuffle': False,
                         'num_workers': 1}
    train_set = DatasetNuscenesCalibNet(**args_train_set)    
    train_loader = torch.utils.data.DataLoader(train_set, **args_train_loader)
    data_iterator = enumerate(train_loader)
    sample = train_set[0]
    #%%
    for i, sample in enumerate(train_loader):
        if i != 0:
            continue
        
        im1 = sample['rgb'][0].cpu().numpy().transpose((1,2,0))
        # mis_radar_depth_map = sample['depth_img'][0].cpu().numpy().transpose((1,2,0))
        # gt = sample['depth_gt'][0].cpu().numpy().transpose((1,2,0)).squeeze()
        
        pc_lidar = sample['point_cloud'][0].clone()
        depth_gt, uv_gt, pc_gt_valid = lidar_project_depth(pc_lidar, sample['calib'][0], (192, 400))  # image_shape
        depth_gt /= 200
        print(depth_gt.shape)
        # print(im1.shape, mis_radar_depth_map.shape, gt.shape)
        plt_im(im1)
        # plt_depth(mis_radar_depth_map)
        # plt_depth_on_im(gt, im1)
        # plt_depth_on_im(mis_radar_depth_map.squeeze(), im1)
        break
    
# %%
