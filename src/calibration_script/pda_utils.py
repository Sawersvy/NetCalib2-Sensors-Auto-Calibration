import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy


class neighbor_connection:   
    def __init__(self, left, right, top, bottom):
        self.xy, self.hn = self.getXYoffset(left, right, top, bottom)
            
           
    def getXYoffset(self, left, right, top, bottom):
        xy = []    
        for x in range(-left, right + 1):
            for y in range(-top, bottom + 1):
                xy.append([x,y])
        
        hn = max([left, right, top, bottom])
        
        return xy, hn
    
    def reflect(self):        
        self.xy = [[-x,-y] for x,y in self.xy]
        
    def plot_neighbor(self):
        xy = self.xy
        hn = self.hn
       
        M = np.zeros((2*hn + 1, 2*hn + 1), dtype=np.uint8)   
        for x, y in xy:
            x += hn
            y += hn
            M[y,x]=255
        M[hn, hn] = 128
        
        plt.imshow(M, cmap='gray')
        plt.title('%d neighbors' % len(xy))
        plt.show()


def makeOffsetConv(xy, hn):
    """Create a 2D convolution that does a separate offset for
    each element in xy"""
    m = nn.Conv2d(1,len(xy),2*hn+1, padding=hn, bias=False)
    m.weight.data.fill_(0)
    for ind, xyo in enumerate(xy):
        m.weight.data[ind,0,hn+xyo[1],hn+xyo[0]] = 1     # weight size [len(xy), 1, 2*hn+1, 2*hn+1]
    return m


def isConnected(d_radar, d_lidar, cfilter):
    
    cshape = d_radar.shape
    if len(cshape)==2:
        nshape = (1,1,cshape[0],cshape[1])
    elif len(cshape)==4:        
        nshape = cshape
    else:
        assert(False)   
    
    d_radar = d_radar.reshape( nshape )
    offsets = cfilter( d_lidar )            
    connection = -torch.ones_like(offsets)

    rel_error = 0.05
    abs_error = 1
    msk_overlap = (d_radar > 0) & (offsets > 0) 

    connection[  msk_overlap & ( torch.abs(d_radar - offsets) < abs_error ) & ( torch.abs(d_radar - offsets)/offsets < rel_error  ) ] = 1
    connection[  msk_overlap & ( (torch.abs(d_radar - offsets) >= abs_error) | ( torch.abs(d_radar - offsets)/offsets >= rel_error ) ) ] = 0
        
    return connection


def depth_to_connect(d_radar, d_lidar, neighbor, device):
    
    xy, hn = neighbor.xy, neighbor.hn  
    cfilter = makeOffsetConv(xy, hn).to(device)    
    connected = isConnected(d_radar, d_lidar, cfilter)  
    
    return connected


def cal_nb_depth(d_radar, neighbor):
    '''
    Get depth in the neighboring region
    
    input:
        d_radar: h x w or n x 1 x h x w
    output:
        nb_depth: 1 x n_nb x h x w
    '''    
    
    if len(d_radar.shape) == 2:
        d_radar = d_radar[None, None, ...]
    
    xy, hn = neighbor.xy, neighbor.hn  
    cfilter = makeOffsetConv(xy, hn).cuda()
    with torch.no_grad():
        nb_depth = cfilter( d_radar )
    
    return nb_depth



def otherHalf(connection, xy):
    """Return other half of connections for each pixel
       Can concatenate this with makeConnections output to get all connections for each pixel, see allConnections()
    """
    assert(len(xy)==connection.shape[1]) #should be one xy offset per connection
    #other = -torch.ones_like(connection)  #if want to say unknown connection to neighbors, need to make sure padding is -1 (rather than zero-padding) in isConnected
    other = torch.zeros_like(connection)
    for ind, xyo in enumerate(xy):
        if xyo[0]==0:
            if xyo[1]==0:
                other[:,ind,:,:] = connection[:,ind,:,:]  #This one is never called as we don't do pixels to each other
            elif xyo[1]<0:
                other[:,ind,:xyo[1],:] = connection[:,ind,-xyo[1]:,:]
            else:
                other[:,ind,xyo[1]:,:] = connection[:,ind,:-xyo[1],:]
        elif xyo[0]<0:
            if xyo[1]==0:
                other[:,ind,:,:xyo[0]] = connection[:,ind,:,-xyo[0]:]
            elif xyo[1]<0:
                other[:,ind,:xyo[1],:xyo[0]] = connection[:,ind,-xyo[1]:,-xyo[0]:]
            else:
                other[:,ind,xyo[1]:,:xyo[0]] = connection[:,ind,:-xyo[1],-xyo[0]:]
        else:
            if xyo[1]==0:
                other[:,ind,:,xyo[0]:] = connection[:,ind,:,:-xyo[0]]
            elif xyo[1]<0:
                other[:,ind,:xyo[1],xyo[0]:] = connection[:,ind,-xyo[1]:,:-xyo[0]]
            else:
                other[:,ind,xyo[1]:,xyo[0]:] = connection[:,ind,:-xyo[1],:-xyo[0]]
    return other

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
    
def plt_depth_on_im(depth_map, im, title = '',  ptsSize = 1, path = None):
    
    h,w = im.shape[0:2]    
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    msk = depth_map > 0
    
    plt.figure()
    plt.imshow(im) 
    plt.scatter(x_map[msk], y_map[msk], c=depth_map[msk], s=ptsSize, cmap='jet')
    plt.title(title)
    plt.axis('off')
    if path:
        plt.savefig(path)
    
def save_depth_on_im(depth_map, im, path, title = '',  ptsSize = 1, zoom = 1.0):
    
    h,w = im.shape[0:2]    
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    msk = depth_map > 0
    
    plt.figure()
    plt.scatter(x_map[msk], y_map[msk], c=depth_map[msk], s=ptsSize, cmap='jet')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
    plt.savefig(path + '.png', bbox_inches='tight', pad_inches=0, dpi=100 * zoom)
    plt.close() 
    
def plot_pda(im, prd_aff, d_radar, d_lidar, nb, thres_aff=0.5):
    '''
    input:
        prd_aff: tensor: (n_batch, n_nb, h, w) 
        d_radar: tensor: (n_batch, 1, h, w)
    output:
        d_est: numpy (h,w)
        aff_max: numpy (h,w); the maximum affinity associated with the depth
    
    '''    
    nb_aff = otherHalf(prd_aff, nb.xy)
    nb2 = copy.deepcopy(nb)
    nb2.reflect()

    # print(nb_aff, nb2.shape, nb2)

    print(d_radar, nb2)
    nb_depth = cal_nb_depth(d_radar, nb2)
    
    nb_aff[nb_aff <= thres_aff] = 0
    nb_aff[nb_depth == 0] = 0
    
    # print(nb_aff.shape)
    
    aff_max, _ = torch.max(nb_aff, dim=1, keepdim=True)    
    msk_max = ( aff_max.eq( nb_aff ) ) & (aff_max>0)
    
    # print(aff_max.shape)
    
    n_max = torch.sum(msk_max, dim=1)
    n_max[n_max==0] = -1
    # print(n_max.shape)
    
    # d_est = torch.sum( nb_depth * msk_max, dim=1) / n_max  
    d_est = torch.sum( nb_depth * msk_max, dim=1) 
    # print(d_est.shape)
    
    # print(im.shape)
    
    image = im[0:3].cpu().numpy().transpose((1,2,0))
    d_radar = d_radar[0].cpu().numpy().transpose((1,2,0)).squeeze()
    d_lidar = d_lidar[0].cpu().numpy().transpose((1,2,0)).squeeze()
    pda = n_max.cpu().numpy().transpose((1,2,0)).squeeze()
    d_est = d_est.cpu().numpy().transpose((1,2,0)).squeeze()
    aff_max = aff_max.squeeze().cpu().detach().numpy()
    print(d_radar.shape, pda.shape, image.shape, d_est.shape, aff_max.shape, nb_aff.shape, n_max.shape)
    plt_depth_on_im(pda.squeeze(), image)
    plt_depth_on_im(d_est.squeeze(), image)
    plt_depth_on_im(d_radar.squeeze(), image)
    plt_depth_on_im(d_lidar.squeeze(), image)
    plt_depth_on_im(aff_max.squeeze(), image)
    
def get_aff_max(prd_aff, d_radar, nb, thres_aff=0.5):
    '''
    input:
        prd_aff: tensor: (n_batch, n_nb, h, w) 
        d_radar: tensor: (n_batch, 1, h, w)
    output:
        d_est: numpy (h,w)
        aff_max: numpy (h,w); the maximum affinity associated with the depth
    
    '''    
    nb_aff = otherHalf(prd_aff, nb.xy)
    nb2 = copy.deepcopy(nb)
    nb2.reflect()
    nb_depth = cal_nb_depth(d_radar, nb2)
    nb_aff[nb_aff <= thres_aff] = 0
    nb_aff[nb_depth == 0] = 0
    aff_max, _ = torch.max(nb_aff, dim=1, keepdim=True)    
    msk_max = ( aff_max.eq( nb_aff ) ) & (aff_max>0)
    n_max = torch.sum(msk_max, dim=1)
    n_max[n_max==0] = -1
    d_est = torch.sum( nb_depth * msk_max, dim=1) 
    
    return aff_max, d_est