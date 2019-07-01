import os
import numpy as np
import struct
from torch.utils.data import Dataset
#from scipy.misc import imresize
from PIL import Image
#from .v2v_util import V2VVoxelization

import torchvision.transforms.functional as TF
import math
import collections
import torch
#from torchsample.transforms import *
from scipy.misc import imrotate
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d


CameraConfig = collections.namedtuple('CameraConfig', 'fx,fy,cx,cy,w,h')


'''_pro: perspective transformation
   _bpro: back perspective transformation
'''
# fx, fy, cx, cy, w, h
# 0,  1,  2,  3,  4, 5

"""                      x * focx / z + center_x      y * focy / z + center_y      z   """
_pro = lambda pt3, cfg: [pt3[0]*cfg[0]/pt3[2]+cfg[2], pt3[1]*cfg[1]/pt3[2]+cfg[3], pt3[2]]

"""                       (x - center_x) * z / focx      (y - center_y) * z / focy      z   """
_bpro = lambda pt2, cfg: [(pt2[0]-cfg[2])*pt2[2]/cfg[0], (pt2[1]-cfg[3])*pt2[2]/cfg[1], pt2[2]] 

_loc = lambda pt1, cfg: [pt1[0] + (cfg[2]-cfg[4]/2.)*pt1[2]/cfg[0], pt1[1] + (cfg[3]-cfg[5]/2.)*pt1[2]/cfg[1], pt1[2]]

def xyz2uvd(xyz, cfg):
    '''xyz: list of xyz points
    cfg: camera configuration
    '''
    xyz = xyz.view(-1,3)
    # perspective projection function
    uvd = [_pro(pt3, cfg) for pt3 in xyz]
    return np.array(uvd)

def uvd2xyz(uvd, cfg):
    '''uvd: list of uvd points
    cfg: camera configuration
    '''
    uvd = uvd.view(-1,3)
    # backprojection
    xyz = [_bpro(pt2, cfg) for pt2 in uvd]
    return np.array(xyz)

def xyz2uvd_op(xyz_pts, cfg):
    '''xyz_pts: tensor of xyz points
       camera_cfg: constant tensor of camera configuration
    '''
    xyz_pts = np.reshape(xyz_pts, (-1,3))
    xyz_list = xyz_pts.tolist()
    uvd_list = [_pro(pt, cfg) for pt in xyz_list]
    uvd_pts = np.stack(uvd_list)
    return uvd_pts #np.reshape(uvd_pts, (-1,))

def uvd2xyz_op(uvd_pts, cfg):
    uvd_pts = np.reshape(uvd_pts, (-1,3))
    uvd_list = uvd_pts.tolist()
    xyz_list = [_bpro(pt, cfg) for pt in uvd_list]
    xyz_pts = np.stack(xyz_list)
    return xyz_pts #np.reshape(xyz_pts, (-1,))

def xyz2xyz_local(xyz_pts, cfg):
    '''xyz_pts: tensor of xyz points
       camera_cfg: constant tensor of camera configuration
    '''
    xyz_pts = np.reshape(xyz_pts, (-1,3))
    xyz_list = xyz_pts.tolist()
    xyzloc_list = [_loc(pt, cfg) for pt in xyz_list]
    xyzloc_pts = np.stack(xyzloc_list)
    return xyzloc_pts

def center_of_mass(dm, cfg):
    #dm = torch.from_numpy(dm)
    shape = dm.shape
    c_h, c_w = shape[0], shape[1]
    ave_u, ave_v = c_w/2, c_h/2 #tf.cast(c_w/2, tf.float32), tf.cast(c_h/2, tf.float32)
    ave_d = (dm * (dm > 0).astype(float)).mean() #tf.reduce_mean(tf.boolean_mask(dm, tf.greater(dm,0)))

    ave_d = ave_d.clip(200.0)

    ave_x = (ave_u-cfg[2])*ave_d/cfg[0]
    ave_y = (ave_v-cfg[3])*ave_d/cfg[1]
    ave_xyz=np.stack([ave_x,ave_y,ave_d])
    return ave_xyz

def crop_from_xyz_pose(dm, pose, cfg, out_w, out_h, max_depth, pad=20.0):
    '''crop depth map by generate the bounding box according to the pose
    Args:
        dms: depth map
        poses: either estimated or groundtruth in xyz coordinate
        cfg: the initial camera configuration
        out_w: output width
        out_h: output height
    Returns:
        crop_dm: the cropped depth map
        new_cfg: the new camera configuration for the cropped depth map
    '''
    #with tf.name_scope('crop'):
        # determine bouding box from pose
    in_h, in_w = dm.shape[0], dm.shape[1] #dm.get_shape()[0].value, dm.get_shape()[1].value
    uvd_pose = np.reshape(xyz2uvd_op(pose,cfg), (-1,3)) #tf.reshape(xyz2uvd_op(pose,cfg), (-1,3))
    min_coor = np.min(uvd_pose, axis=0)
    max_coor = np.max(uvd_pose, axis=0)

    top = np.minimum(np.maximum(min_coor[1]-pad, 0.0), cfg.h-2*pad)
    left = np.minimum(np.maximum(min_coor[0]-pad, 0.0), cfg.w-2*pad)
    bottom = np.maximum(np.minimum(max_coor[1]+pad, cfg.h), top.astype(float)+2*pad-1)
    right = np.maximum(np.minimum(max_coor[0]+pad, cfg.w), left.astype(float)+2*pad-1)

    top = top.astype(int)
    left = left.astype(int)
    bottom = bottom.astype(int)
    right = right.astype(int)

    cropped_dm = dm[dm.shape[0]-bottom:dm.shape[0]-top,left:right]
    #cropped_dm = tf.image.crop_to_bounding_box(dm,
    #                                          offset_height=top,
    #                                          offset_width=left,
    #                                          target_height=bottom-top,
    #                                          target_width=right-left)

    longer_edge = np.maximum(bottom-top, right-left)
    offset_height = (longer_edge-bottom+top) // 2 #tf.to_int32(tf.divide(longer_edge-bottom+top, 2))
    offset_width = (longer_edge-right+left) // 2 #tf.to_int32(tf.divide(longer_edge-right+left, 2))
    
    cropped_dm = np.concatenate([max_depth*np.ones((offset_height, cropped_dm.shape[1])), cropped_dm, max_depth*np.ones((longer_edge-cropped_dm.shape[0]-offset_height, cropped_dm.shape[1]))], axis=0)
    cropped_dm = np.concatenate([max_depth*np.ones((cropped_dm.shape[0], offset_width)), cropped_dm, max_depth*np.ones((cropped_dm.shape[0], longer_edge-cropped_dm.shape[1]-offset_width))], axis=1)

    #cropped_dm = tf.image.pad_to_bounding_box(cropped_dm,
    #                                         offset_height=offset_height,
    #                                         offset_width=offset_width,
    #                                         target_height=longer_edge,
    #                                         target_width=longer_edge)
    #cropped_dm = tf.image.resize_images(cropped_dm, (out_h, out_w))
    cropped_dm = np.array(Image.fromarray(cropped_dm).resize((out_h, out_w)))

    # to further earse the background
    #uvd_list = uvd_pose.transpose().tolist() #tf.unstack(uvd_pose, axis=-1)

    uu = uvd_pose[:,0].astype(int).clip(0, in_w-1)
    vv = uvd_pose[:,1].astype(int).clip(0, in_h-1)
    dd = np.stack([dm[vv[i], uu[i]] for i in range(uu.shape[0])])
    dd = (dd>100).astype(float) * dd
    d_th = np.min(dd) + 250.0
    
    cropped_dm = np.where(cropped_dm < d_th, cropped_dm, np.ones_like(cropped_dm)*max_depth) #tf.less(cropped_dm,d_th), cropped_dm, tf.zeros_like(cropped_dm))


    #uu = tf.clip_by_value(tf.to_int32(uvd_list[0]), 0, in_w-1)
    #vv = tf.clip_by_value(tf.to_int32(uvd_list[1]), 0, in_h-1)

    #dd = tf.gather_nd(dm, tf.stack([vv,uu], axis=-1))
    #dd = tf.boolean_mask(dd, tf.greater(dd, 100))
    #d_th = tf.reduce_min(dd) + 250.0
    #if FLAGS.dataset == 'icvl':
    #    cropped_dm = tf.where(tf.less(cropped_dm,500.0), cropped_dm, tf.zeros_like(cropped_dm))
    #else:
    #    cropped_dm = tf.where(tf.less(cropped_dm,d_th), cropped_dm, tf.zeros_like(cropped_dm))

    #with tf.name_scope('cfg'):
    ratio_x = longer_edge/out_w
    ratio_y = longer_edge/out_h
    top = top.astype(float)
    left = left.astype(float)

    new_cfg = np.stack([cfg.fx/ratio_x, cfg.fy/ratio_y, 
                        (cfg.cx-left+offset_width.astype(float))/ratio_x, 
                        (cfg.cy-top+offset_height.astype(float))/ratio_y,
                        float(out_w), float(out_h)], axis=0) 

    return [cropped_dm, pose, new_cfg]

def image_resize_with_crop_or_pad(dm, shape):
    pad = (shape[0] - dm.shape[0])
    pad_1 = pad // 2
    pad_2 = pad - pad_1
    if pad > 0:
        resized_dm = np.concatenate([np.zeros((pad_1, dm.shape[1])), dm, np.zeros((pad_2, dm.shape[1]))], axis=0)
    else:
        resized_dm = dm[-pad_1:dm.shape[0]+pad_2, :]
    pad = (shape[1] - resized_dm.shape[1])
    pad_1 = pad // 2
    pad_2 = pad - pad_1
    if pad > 0:
        resized_dm = np.concatenate([np.zeros((resized_dm.shape[0], pad_1)), resized_dm, np.zeros((resized_dm.shape[0], pad_2))], axis=1)
    else:
        resized_dm = resized_dm[:, -pad_1:resized_dm.shape[1]+pad_2]
    return resized_dm

def data_aug(dm, pose, cfg, com):
    # random rotation
    jnt_num = pose.shape[0]
    angle = np.random.rand(1)*2.-1
    rot_dm = np.array(Image.fromarray(dm).rotate(-180.*angle))
    angle *= math.pi

    uv_com = xyz2uvd_op(com, cfg)
    uvd_pt = xyz2uvd_op(pose, cfg) - np.tile(uv_com,[jnt_num, 1])
    cost, sint = np.cos(angle)[0], np.sin(angle)[0]
    rot_mat = np.stack([cost,-sint,0, sint,cost,0, 0.0,0.0,1.0], axis=0)
    rot_mat = np.reshape(rot_mat, (3,3))

    #uvd_pt = np.reshape(uvd_pt, (-1,3))
    rot_pose = np.reshape(np.matmul(uvd_pt, rot_mat), (-1,))
   
    # random elongate x,y edge
    edge_ratio = np.clip(np.random.randn(2)*0.2+1.0, 0.9, 1.1)
    target_height = (dm.shape[0]*edge_ratio[0]).astype(int)  #target_height = tf.to_int32(tf.to_float(tf.shape(dm)[0])*edge_ratio[0])
    target_width = (dm.shape[1]*edge_ratio[1]).astype(int)   #target_width = tf.to_int32(tf.to_float(tf.shape(dm)[1])*edge_ratio[1])
    # 1 stands for nearest neighour interpolation
    rot_dm = np.array(Image.fromarray(rot_dm).resize((target_height, target_width)))  #rot_dm = tf.image.resize_images(rot_dm, (target_height, target_width), 1)

    rot_dm = image_resize_with_crop_or_pad(rot_dm, dm.shape)  #rot_dm = tf.image.resize_image_with_crop_or_pad(rot_dm, tf.shape(dm)[0], tf.shape(dm)[1])
    rot_pose = rot_pose * np.tile([edge_ratio[1],edge_ratio[0],1.0], [jnt_num])

    rot_pose = rot_pose + np.tile(uv_com, [jnt_num])
    rot_pose = uvd2xyz_op(rot_pose, cfg)
    rot_pose = np.reshape(rot_pose, pose.shape)
    return rot_dm, rot_pose



def pixel2world(x, y, z, img_width, img_height, fx, fy):
    w_x = (x - img_width / 2) * z / fx
    w_y = (img_height / 2 - y) * z / fy
    w_z = z
    return w_x, w_y, w_z


def world2pixel(x, y, z, img_width, img_height, fx, fy):
    p_x = x * fx / z + img_width / 2
    p_y = img_height / 2 - y * fy / z
    return p_x, p_y


def depthmap2points(image, fx, fy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, w, h, fx, fy)
    return points


def points2pixels(points, img_width, img_height, fx, fy):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        world2pixel(points[:,0], points[:, 1], points[:, 2], img_width, img_height, fx, fy)
    return pixels


def load_depthmap(filename, img_width, img_height, max_depth):
    with open(filename, mode='rb') as f:
        data = f.read()
        _, _, left, top, right, bottom = struct.unpack('I'*6, data[:6*4])
        num_pixel = (right - left) * (bottom - top)
        cropped_image = struct.unpack('f'*num_pixel, data[6*4:])

        cropped_image = np.asarray(cropped_image).reshape(bottom-top, -1)
        depth_image = np.zeros((img_height, img_width), dtype=np.float32)
        depth_image[top:bottom, left:right] = cropped_image
        depth_image[depth_image == 0] = max_depth
        
        """
        max = depth_image.max()
        depth_image[depth_image == 0] = max_depth
        min = depth_image.min()
        depth_image[depth_image == max_depth] = min
        depth_image -= min
        """

        return depth_image


class MARAHandDataset(Dataset):
    approximate_num_per_file = 220 
    name = 'icvl'
    #max_depth = 500.0
    pose_dim = 48
    jnt_num = 16
    
    
    def __init__(self, root, mode, test_subject_id, transform=None):
        self.img_width = 320
        self.img_height = 240
        self.min_depth = 100
        self.max_depth = 700
        self.fx = 241.42
        self.fy = 241.42
        self.joint_num = 21
        self.world_dim = 3
        self.folder_list = ['1','2','3','4','5','6','7','8','9','I','IP','L','MP','RP','T','TIP','Y']
        self.subject_num = 9

        self.cfg = CameraConfig(fx=self.fx, fy=self.fy, cx=self.img_width/2, cy=self.img_height/2, w=self.img_width, h=self.img_height)

        self.stored_max = np.array([-1000.,-1000.,-1000.])
        self.stored_min = -self.stored_max

        self.root = root
        #self.center_dir = center_dir
        self.mode = mode
        self.test_subject_id = test_subject_id
        self.transform = transform

        if not self.mode in ['train', 'test']: raise ValueError('Invalid mode')
        assert self.test_subject_id >= 0 and self.test_subject_id < self.subject_num

        self.training = (mode == 'train')

        if not self._check_exists(): raise RuntimeError('Invalid MSRA hand dataset')
        
        self._load()
        
        """ NORMALIZE """
        #coords = []
        
        #MIN, MAX = [-226.25311419 -248.34654969 -443.61077552] [1394.25539992 1502.09045378  564.196     ]
        min = np.array([-110., -110., 0.])
        max = np.array([110., 110., 600.])
        self.offset = -min - (max-min)/2
        #self.scale = 100.0
        self.scale = (max-min) / 2
        
        """        
        self.scale = 0. #[1.,1.,1.]
        self.offset = np.zeros(3)
        for i in range(3):
            min_a = self.joints_world[:,:,i].min()
            min_b = self.joints_world_alt[:,:,i].min()
            max_a = self.joints_world[:,:,i].max()
            max_b = self.joints_world_alt[:,:,i].max()
            min = min_a if min_a < min_b else min_b
            max = max_a if max_a > max_b else max_b
            self.offset[i] = -min - (max-min)/2
            scale = (max-min)/1.8
            if scale > self.scale:
                self.scale = scale

        self.scale *= 6.max_depth
        """


        """
        offset = np.tile(offset[None,None,:], (self.joints_world.shape[0], self.joints_world.shape[1], 1))
        self.joints_world_normalized = self.joints_world + offset

        scale = np.tile(np.array([self.scale])[None,None,:], self.joints_world.shape)
        self.joints_world_normalized = self.joints_world_normalized / scale
        
        wrist = self.joints_world_normalized[:,0,:][:,None,:]

        self.joints_world_normalized = np.concatenate([wrist, self.joints_world_normalized[:,1:5,:],
                                                       wrist, self.joints_world_normalized[:,5:9,:],
                                                       wrist, self.joints_world_normalized[:,9:13,:],
                                                       wrist, self.joints_world_normalized[:,13:17,:],
                                                       wrist, self.joints_world_normalized[:,17:,:]
                                                       ], axis=1)

        
        self.joints_world_normalized = self.joints_world_normalized.reshape(self.joints_world_normalized.shape[0], 5, -1, 3)
        self.joints_world_normalized = self.joints_world_normalized.reshape(self.joints_world_normalized.shape[0], 5, -1)
        """
        
        self.not_initialized = True

    def test(self):
        for _ in range(100):
            index = int(torch.rand(1)*20000)
            print(index)
            depthmap_orig = load_depthmap(self.names[index], self.img_width, self.img_height, self.max_depth)

            
            #depthmap = -depthmap_orig + self.max_depth
            pose_orig = self.joints_world[index]
            """

            uvd_pose = xyz2uvd_op(pose_orig, self.cfg)

            data = []
            ysh, xsh = depthmap_orig.shape[0], depthmap_orig.shape[1]
            for y in range(ysh):
                for x in range(xsh):
                    if depthmap_orig[y,x] < 600 or x==0 or x==(xsh-1) or y==0 or y==(ysh-1):
                        if torch.rand(1) > 0.9:
                            point = np.array([x,(ysh-1)-y,depthmap_orig[y,x]])
                            data.append(point)
            data = np.stack(data)
            ax = m3d.Axes3D(plt.figure())
            ax.scatter3D(*data.T, c='r')
            ax.scatter3D(*uvd_pose.T, c='b')
            plt.show()
            """

            cropped_dm, cropped_pose, cropped_cfg = crop_from_xyz_pose(depthmap_orig, pose_orig, self.cfg, out_w=128, out_h=128, pad=20.0, max_depth=self.max_depth)

            #uvd_pose = xyz2uvd_op(cropped_pose, cropped_cfg)

            xyzlocal_pose = xyz2xyz_local(cropped_pose, cropped_cfg)

            """
            data = []
            ysh, xsh = cropped_dm.shape[0], cropped_dm.shape[1]
            for y in range(ysh):
                for x in range(xsh):
                    if cropped_dm[y,x] < 600 or x==0 or x==(xsh-1) or y==0 or y==(ysh-1):
                        if torch.rand(1) > 0.9:
                            point = np.array([x,(ysh-1)-y,cropped_dm[y,x]])
                            data.append(point)
            data = np.stack(data)
            ax1 = m3d.Axes3D(plt.figure())
            ax1.scatter3D(*data.T, c='r')
            ax1.scatter3D(*uvd_pose.T, c='b')
            plt.show()
            """

            ax1a = m3d.Axes3D(plt.figure())
            ax1a.scatter3D(*xyzlocal_pose.T, c='b')
            plt.show()
            


            com = center_of_mass(cropped_dm, cropped_cfg)
            inv_depthmap = -cropped_dm + self.max_depth
            aug_dms, pose = data_aug(inv_depthmap, cropped_pose, cropped_cfg, com)
            depthmap = -aug_dms + self.max_depth



            """
            uvd_pose = xyz2uvd_op(pose,cropped_cfg)
    
            
            data = []
            ysh, xsh = depthmap.shape[0], depthmap.shape[1]
            for y in range(ysh):
                for x in range(xsh):
                    if depthmap[y,x] < 600 or x==0 or x==(xsh-1) or y==0 or y==(ysh-1):
                        if torch.rand(1) > 0.9:
                            point = np.array([x,(ysh-1)-y,depthmap[y,x]])
                            data.append(point)
            data = np.stack(data)
            ax2 = m3d.Axes3D(plt.figure())
            ax2.scatter3D(*data.T, c='r')
            ax2.scatter3D(*uvd_pose.T, c='b')
            """

            xyzlocal_pose = xyz2xyz_local(pose, cropped_cfg)
            ax3 = m3d.Axes3D(plt.figure())
            ax3.scatter3D(*xyzlocal_pose.T, c='b')
            
            plt.show()
            print()


    def __getitem__(self, index):
        depthmap_orig = load_depthmap(self.names[index], self.img_width, self.img_height, self.max_depth)
        pose_orig = self.joints_world[index]

        depthmap, pose, cropped_cfg = crop_from_xyz_pose(depthmap_orig, pose_orig, self.cfg, out_w=128, out_h=128, pad=20.0, max_depth=self.max_depth)

        
        if self.training:
            com = center_of_mass(depthmap, cropped_cfg)
            inv_depthmap = -depthmap + self.max_depth
            aug_dms, pose = data_aug(inv_depthmap, pose, cropped_cfg, com)
            depthmap = -aug_dms + self.max_depth
        

        xyzlocal_pose = xyz2xyz_local(pose, cropped_cfg)



        """
        for i in range(3):
            if pose[:,i].min() < self.stored_min[i]:
                self.stored_min[i] = pose[:,i].min()
                print()
                print('MIN, MAX =', self.stored_min, self.stored_max)
    
            if pose[:,i].max() > self.stored_max[i]:
                self.stored_max[i] = pose[:,i].max()
                print()
                print('MIN, MAX =', self.stored_min, self.stored_max)
        """


        offset = np.tile(self.offset[None,:], (xyzlocal_pose.shape[0], 1))
        joints_world_normalized = xyzlocal_pose + offset

        #scale = np.tile(np.array([self.scale])[None,:], aug_poses.shape)
        joints_world_normalized = joints_world_normalized / self.scale
        
        if joints_world_normalized.min() < -1 or joints_world_normalized.max() > 1:
            print('trouble trouble trouble ', joints_world_normalized.min(), joints_world_normalized.max())
            
        
        wrist = joints_world_normalized[0,:][None,:]

        joints_world_normalized = np.concatenate([wrist, joints_world_normalized[1:5,:],
                                                       wrist, joints_world_normalized[5:9,:],
                                                       wrist, joints_world_normalized[9:13,:],
                                                       wrist, joints_world_normalized[13:17,:],
                                                       wrist, joints_world_normalized[17:,:]
                                                       ], axis=0)

        joints_world_normalized = joints_world_normalized.reshape(5, -1, 3)
        joints_world_normalized = joints_world_normalized.reshape(5, -1)




        if self.not_initialized:
            self.not_initialized = False
            """
            points = depthmap2points(depthmap, self.fx, self.fy)
            points = points.reshape((-1, 3))

            j = 0
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca(projection='3d')
            for i in range(points.shape[0]):
                xs, ys, zs = points[i,:]
                if zs != self.max_depth:
                    if (j % 10) == 0:
                        ax.scatter(xs, ys, zs, c='r', marker='o')
                    j += 1
            for i in range(self.joints_world[index].shape[0]):
                xs, ys, zs = self.joints_world[index][i,:]
                ax.scatter(xs, ys, zs, c='b', marker='o')
            #plt.savefig('fig_{}.png'.format(i), dpi=400, bbox_inches='tight')
            plt.show()
            print('hej')
            """

        depthmap /= self.max_depth
        depthmap = 1 - depthmap

        #depthmap = np.concatenate([depthmap, np.zeros((self.img_width-self.img_height, self.img_width), dtype=np.float32)], axis=0)
        
        #depthmap = np.array(Image.fromarray(depthmap).resize((100, 100)))
        #depthmap = imresize(depthmap, (100,100), interp='bilinear', mode='F')
        
        
        return np.float32(depthmap.reshape((1, *depthmap.shape))), np.float32(joints_world_normalized)
            #save_to_jpg('test%1.png', depthmap, format="PNG")
        """
        sample = {
            'name': self.names[index],
            #'points': points,
            'joints': self.joints_world[index],
            'refpoint': self.ref_pts[index],
            'depthmap': depthmap
        }

        if self.transform: sample = self.transform(sample)

        return sample
        """

    def __len__(self):
        return self.num_samples

    def _load(self):
        self._compute_dataset_size()

        if self.mode == 'train':
            self.num_samples = self.train_size
            self.num_samples_alt = self.test_size
        else:
            self.num_samples = self.test_size
            self.num_samples_alt = self.train_size

        #self.num_samples = self.train_size if self.mode == 'train' else self.test_size
        self.joints_world = np.zeros((self.num_samples, self.joint_num, self.world_dim))
        self.joints_world_alt = np.zeros((self.num_samples_alt, self.joint_num, self.world_dim))
        #self.ref_pts = np.zeros((self.num_samples, self.world_dim))
        self.names = []

        # Collect reference center points strings
        #if self.mode == 'train': ref_pt_file = 'center_train_' + str(self.test_subject_id) + '_refined.txt'
        #else: ref_pt_file = 'center_test_' + str(self.test_subject_id) + '_refined.txt'

        #with open(os.path.join(self.center_dir, ref_pt_file)) as f:
        #        ref_pt_str = [l.rstrip() for l in f]

        #
        #file_id = 0
        frame_id = 0
        frame_id_alt = 0

        for mid in range(self.subject_num):
            if self.mode == 'train': model_chk = (mid != self.test_subject_id)
            elif self.mode == 'test': model_chk = (mid == self.test_subject_id)
            else: raise RuntimeError('unsupported mode {}'.format(self.mode))
            
            #if model_chk:
            for fd in self.folder_list:
                annot_file = os.path.join(self.root, 'P'+str(mid), fd, 'joint.txt')

                lines = []
                with open(annot_file) as f:
                    lines = [line.rstrip() for line in f]

                # skip first line
                for i in range(1, len(lines)):
                    # referece point
                    #splitted = ref_pt_str[file_id].split()
                    #if splitted[0] == 'invalid':
                    #    print('Warning: found invalid reference frame')
                    #    file_id += 1
                    #    continue
                    #else:
                    #    self.ref_pts[frame_id, 0] = float(splitted[0])
                    #    self.ref_pts[frame_id, 1] = float(splitted[1])
                    #    self.ref_pts[frame_id, 2] = float(splitted[2])

                    # joint point
                    splitted = lines[i].split()
                    if model_chk:
                        for jid in range(self.joint_num):
                            self.joints_world[frame_id, jid, 0] = float(splitted[jid * self.world_dim])
                            self.joints_world[frame_id, jid, 1] = float(splitted[jid * self.world_dim + 1])
                            self.joints_world[frame_id, jid, 2] = -float(splitted[jid * self.world_dim + 2])
                        frame_id += 1
                        filename = os.path.join(self.root, 'P'+str(mid), fd, '{:0>6d}'.format(i-1) + '_depth.bin')
                        self.names.append(filename)
                    else:
                        for jid in range(self.joint_num):
                            self.joints_world_alt[frame_id_alt, jid, 0] = float(splitted[jid * self.world_dim])
                            self.joints_world_alt[frame_id_alt, jid, 1] = float(splitted[jid * self.world_dim + 1])
                            self.joints_world_alt[frame_id_alt, jid, 2] = -float(splitted[jid * self.world_dim + 2])
                        frame_id_alt += 1

                    #file_id += 1

    def _compute_dataset_size(self):
        self.train_size, self.test_size = 0, 0

        for mid in range(self.subject_num):
            num = 0
            for fd in self.folder_list:
                annot_file = os.path.join(self.root, 'P'+str(mid), fd, 'joint.txt')
                with open(annot_file) as f:
                    num = int(f.readline().rstrip())
                if mid == self.test_subject_id: self.test_size += num
                else: self.train_size += num

    def _check_exists(self):
        # Check basic data
        for mid in range(self.subject_num):
            for fd in self.folder_list:
                annot_file = os.path.join(self.root, 'P'+str(mid), fd, 'joint.txt')
                if not os.path.exists(annot_file):
                    print('Error: annotation file {} does not exist'.format(annot_file))
                    return False

        # Check precomputed centers by v2v-hand model's author
        """
        for subject_id in range(self.subject_num):
            center_train = os.path.join(self.center_dir, 'center_train_' + str(subject_id) + '_refined.txt')
            center_test = os.path.join(self.center_dir, 'center_test_' + str(subject_id) + '_refined.txt')
            if not os.path.exists(center_train) or not os.path.exists(center_test):
                print('Error: precomputed center files do not exist')
                return False
        """

        return True
    
    
"""
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
def plotXYZ(XYZ, max_depth):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection='3d')
    for i in range(XYZ.shape[0]):
        #for j in range(XYZ.shape[1]):
        xs, ys, zs = XYZ[i,:]
        if zs != max_depth:
            ax.scatter(xs, ys, zs, c='r', marker='o')
#     plt.savefig('fig_{}.png'.format(i), dpi=400, bbox_inches='tight')
    plt.show()

from PIL import Image
def save_to_jpg(filename, array, format="PNG"):
    #filename = "dump_1.png"
    #os.chdir(r"./dumps")
    while True:
        file = ''
        for file in os.listdir(os.curdir):
            if file == filename:
                name, ext = file.split(".")
                word, number = name.split("%")
                new_num = int(number) + 1
                filename = word + "%" + str(new_num) + "." + ext
                file = filename
                break
            else:
                continue
        if (file != filename):
            break

    image = Image.fromarray(array)
    image = image.convert("L")
    image.save(filename, format)
"""



def foo(points, verbose=False):
    stk = []
    for i in range(points.shape[0]):
        data = points[i, [1,5,9,13],:]
        rest = points[i, [0,2,3,4,6,7,8,10,11,12,14,15,16,17,18,19,20],:]
        
        # Calculate the mean of the points, i.e. the 'center' of the cloud
        datamean = data.mean(axis=0)
        
        # Do an SVD on the mean-centered data.
        uu, dd, vv = np.linalg.svd(data - datamean)
        
        x = datamean - points[i, 0]
        z = np.cross(x, vv[0])
        z = z/np.linalg.norm(z)
        x = x/np.linalg.norm(x)
        y = np.cross(z,x)
    
        trans = np.array([x,y,z,points[i, 0]])
        trans = np.concatenate([trans, np.zeros((4,1))], axis=1).T
        trans[3,3] = 1
        invTrans = np.linalg.inv(trans)
        stk.append(trans[:3, 0])
        stk.append(trans[:3, 1])
        stk.append(trans[:3, 3])
        stk.append(np.zeros(3))
        
        hand = points[i,1:].reshape(5,4,3)
        for finger in range(hand.shape[0]):
            for j in range(hand.shape[1]):
                p = np.array((hand[finger,j,0], hand[finger,j,1], hand[finger,j,2], 1))
                p_local = np.matmul(invTrans, p)[:3]
                if j>0:
                    p_delta = p_local - p_local_old
                    stk.append(p_delta)
                else:
                    stk.append(p_local)
                p_local_old = p_local

        if verbose:
            linex = x * np.mgrid[0.:0.2:2j][:, np.newaxis] + points[0]
            liney = y * np.mgrid[0.:0.2:2j][:, np.newaxis] + points[0]
            linez = z * np.mgrid[0.:0.2:2j][:, np.newaxis] + points[0]
            # Now vv[0] contains the first principal component, i.e. the direction
            # vector of the 'best fit' line in the least squares sense.
            
            # Now generate some points along this best fit line, for plotting.
            
            # I use -7, 7 since the spread of the data is roughly 14
            # and we want it to have mean 0 (like the points we did
            # the svd on). Also, it's a straight line, so we only need 2 points.
            linepts = vv[0] * np.mgrid[-0.2:0.2:2j][:, np.newaxis]
            
            # shift by the mean to get the line in the right place
            linepts += datamean
            
            # Verify that everything looks right.
            ax = m3d.Axes3D(plt.figure())
            ax.scatter3D(*data.T, c='r')
            ax.scatter3D(*datamean.T, c='g')
            ax.scatter3D(*rest.T, c='b')
            ax.plot3D(*linepts.T)
            ax.plot3D(*linex.T, c='r')
            ax.plot3D(*liney.T, c='g')
            ax.plot3D(*linez.T, c='b')
            plt.show()
    
    transformed = np.stack(stk, axis=0).reshape(-1, 6, 4*3)
    #print(transformed.shape)
    return transformed
