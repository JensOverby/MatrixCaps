import os
import numpy as np
import struct
from torch.utils.data import Dataset
#from .v2v_util import V2VVoxelization

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


class MARAHandDataset(Dataset):
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

        self.root = root
        #self.center_dir = center_dir
        self.mode = mode
        self.test_subject_id = test_subject_id
        self.transform = transform

        if not self.mode in ['train', 'test']: raise ValueError('Invalid mode')
        assert self.test_subject_id >= 0 and self.test_subject_id < self.subject_num

        if not self._check_exists(): raise RuntimeError('Invalid MSRA hand dataset')
        
        self._load()
        
        """ NORMALIZE """
        coords = []
        for i in range(self.joints_world.shape[2]):
            min_a = self.joints_world[:,:,i].min()
            min_b = self.joints_world_alt[:,:,i].min()
            max_a = self.joints_world[:,:,i].max()
            max_b = self.joints_world_alt[:,:,i].max()
            min = min_a if min_a < min_b else min_b
            max = max_a if max_a > max_b else max_b
            coord = self.joints_world[:,:,i].copy()
            coord -= min
            coord /= ((max-min)/1.8)
            coord -= 0.9
            coords.append(coord)
        self.joints_world_normalized = np.stack(coords, axis=2)
        
        
        #fingers = self.joints_world_normalized[:,1:,:]
        wrist = self.joints_world_normalized[:,0,:][:,None,:]
        #
        #fingers = fingers - np.tile(wrist, (1,20,1))
        self.joints_world_normalized = np.concatenate([self.joints_world_normalized[:,1:5,:], wrist,
                                                       self.joints_world_normalized[:,5:9,:], wrist,
                                                       self.joints_world_normalized[:,9:13,:], wrist,
                                                       self.joints_world_normalized[:,13:17,:], wrist,
                                                       self.joints_world_normalized[:,17:,:], wrist
                                                       ], axis=1)

        #self.joints_world_normalized = np.concatenate([fingers, wrist, np.zeros((fingers.shape[0], 3, 3), dtype=np.float32)], axis=1)
        
        self.joints_world_normalized = self.joints_world_normalized.reshape(self.joints_world_normalized.shape[0], 5, -1)
        
        self.not_initialized = True
    
    def __getitem__(self, index):
        depthmap = load_depthmap(self.names[index], self.img_width, self.img_height, self.max_depth)

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

        #depth_image = np.zeros((self.img_width-self.img_height, self.img_width), dtype=np.float32)
        depthmap = np.concatenate([depthmap, np.zeros((self.img_width-self.img_height, self.img_width), dtype=np.float32)], axis=0)
        
        return depthmap.reshape((1, *depthmap.shape)), np.float32(self.joints_world_normalized[index])
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