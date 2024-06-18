import math
import mathutils
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt

def get_mis_calibrate_matrix(max_t, max_r):
    # add noise
    max_angle = max_r
    rotz = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    roty = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    rotx = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    transl_x = np.random.uniform(-max_t, max_t)
    transl_y = np.random.uniform(-max_t, max_t)
    transl_z = np.random.uniform(-max_t, max_t)
    initial_RT = 0.0
        
    R = mathutils.Euler((rotx, roty, rotz))
    T = mathutils.Vector((transl_x, transl_y, transl_z))
    
    R, T = invert_pose(R,T) # calculating quaternions and translation vectors after inversion
    R, T = torch.tensor(R), torch.tensor(T)
    # get a depth map from a perturbed point cloud
    R_m = mathutils.Quaternion(R).to_matrix()
    R_m.resize_4x4()
    T_m = mathutils.Matrix.Translation(T)
    RT_m = T_m @ R_m
    mis_calibrate_matrix = np.array(RT_m) # covert to numpy
    
    gt_error = [rotx, roty, rotz, transl_x, transl_y, transl_z]
    
    return mis_calibrate_matrix, R, T, gt_error

def invert_pose(R, T):
    """
    Given the 'sampled pose' (aka H_init), we want CMRNet to predict inv(H_init).
    inv(T*R) will be used as ground truth for the network.
    Args:
        R (mathutils.Euler): Rotation of 'sampled pose'
        T (mathutils.Vector): Translation of 'sampled pose'

    Returns:
        (R_GT, T_GT) = (mathutils.Quaternion, mathutils.Vector)
    """
    R = R.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(T)
    RT = T @ R
    RT.invert_safe()
    T_GT, R_GT, _ = RT.decompose()
    return R_GT.normalized(), T_GT

def plt_flow_on_im(flow, im, skip = 3):
    
    h,w = im.shape[0:2] 
    plt.figure(figsize=(15,15))
    plt.imshow(im)
    msk1 = flow[:,:,0] != 0
    msk2 = flow[:,:,1] != 0
    msk = np.logical_or(msk1,msk2)
    
    for i in range(0, h, skip+1):
        for j in range(0, w, skip+1):
            if msk[i,j]:
                dx = flow[i, j, 0]
                dy = flow[i, j, 1]
                if 0 <= i + dy < h and 0 <= j + dx < w:
                    plt.arrow(j, i, dx, dy, length_includes_head=True, width=0.05, head_width=0.5, color='cyan')
                # plt.arrow(j,i, flow[i,j,0], flow[i,j,1], length_includes_head=True, width=0.05, head_width=0.5, color='cyan')

def plt_depth(depth):
    plt.figure(figsize=(15,8))
    plt.imshow(depth)
    plt.colorbar()
    
def plt_depth_on_im(depth_map, im, title = '',  ptsSize = 1):
    
    h,w = im.shape[0:2]    
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    msk = depth_map > 0
    
    plt.figure()
    plt.imshow(im) 
    plt.scatter(x_map[msk], y_map[msk], c=depth_map[msk], s=ptsSize, cmap='jet')
    plt.title(title)
    plt.colorbar()
    
def plt_im(im, title = '',  ptsSize = 1):
    plt.figure()
    plt.imshow(im) 
    plt.title(title)
    
class Resampler:
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        if 'points' in sample:
            sample['points'] = self._resample(sample['points'], self.num)
        else:
            if 'crop_proportion' not in sample:
                src_size, ref_size = self.num, self.num
            elif len(sample['crop_proportion']) == 1:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = self.num
            elif len(sample['crop_proportion']) == 2:
                src_size = math.ceil(sample['crop_proportion'][0] * self.num)
                ref_size = math.ceil(sample['crop_proportion'][1] * self.num)
            else:
                raise ValueError('Crop proportion must have 1 or 2 elements')

            sample['points_src'] = self._resample(sample['points_src'], src_size)
            sample['points_ref'] = self._resample(sample['points_ref'], ref_size)

        return sample

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]
