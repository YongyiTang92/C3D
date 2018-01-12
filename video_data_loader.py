import numpy as np
import os
import torch
import torch.utils.data as data
import scipy.io as io
import pickle
import torchvision.transforms as trans
import random
from PIL import Image


class ucf101_rgb_loader_3d(data.Dataset):
    def __init__(self, data_dir, file_dir, image_size=(112, 112), frames_per_clip=8):
        self.data_dir = data_dir  # data_dir = /home/yongyi/ucf101_train/my_code/data
        self.file_dir = file_dir  # file_dir = /home/local/yongyi/...
        self.image_size = image_size
        self.size_all = [128, 112, 96, 84]  # 4 different length for width and height following TSN.\
        self.frames_per_clip = frames_per_clip  # The number of segmentations for a video.

        with open(os.path.join(self.data_dir, 'train_name.pkl'), 'r') as f:
            self.data_name_list = pickle.load(f)
        with open(os.path.join(self.data_dir, 'train_nFrames.pkl'), 'r') as f:
            self.nFrame_list = pickle.load(f)
        with open(os.path.join(self.data_dir, 'train_label.pkl'), 'r') as f:
            self.label_list = pickle.load(f)
        self.transform = None

    def __getitem__(self, index):
        # Read a list of image by index
        target = self.label_list[index:index + 1, :]
        target = torch.from_numpy(target)  # size: (1, 101) 101 classes for ucf101
        # One image example
        file_name = self.data_name_list[index]

        image_index = random.randint(1, self.nFrame_list[index] - self.frames_per_clip)
        img_list = []
        for i in range(self.frames_per_clip):
            img_dir = os.path.join(self.file_dir, file_name, ('frame' + '%06d' % (image_index + i) + '.jpg'))
            img = Image.open(img_dir).convert('RGB')  # convert to RGB
            img_list.append(img)

        # Perform transform
        # Scale jittering
        width_rand = self.size_all[random.randint(0, 3)]
        height_rand = self.size_all[random.randint(0, 3)]
        crop_size = (height_rand, width_rand)
        transform = trans.Compose([trans.Resize(128),
                                   trans.TenCrop(crop_size)])
        transform2 = trans.Compose([trans.Resize(self.image_size), trans.ToTensor()])

        random_crop_index = random.randint(0, 9)
        # img_tensor = img_tensor[random_crop_index, :]
        img_tensor = torch.cat([transform2(transform(img_tmp)[random_crop_index]) for img_tmp in img_list], 0)
        target = target.squeeze()

        return img, target  # img size: (frames_per_clip, 3, 112, 112); target size: (101)

    def __len__(self):
        return len(self.data_name_list)


class ucf101_rgb_loader_3d_test(data.Dataset):
    def __init__(self, data_dir, file_dir, frames_per_clip=8, image_size=(112, 112)):
        self.data_dir = data_dir  # data_dir = /home/yongyi/ucf101_train/my_code/data
        self.file_dir = file_dir  # file_dir = /home/local/yongyi/...
        self.image_size = image_size
        self.size_all = [128, 112, 96, 84]  # 4 different length for width and height following TSN.\
        self.frames_per_clip = frames_per_clip  # The number of segmentations for a video.

        with open(os.path.join(self.data_dir, 'test_name.pkl'), 'r') as f:
            self.data_name_list = pickle.load(f)
        with open(os.path.join(self.data_dir, 'test_nFrames.pkl'), 'r') as f:
            self.nFrame_list = pickle.load(f)
        with open(os.path.join(self.data_dir, 'test_label.pkl'), 'r') as f:
            self.label_list = pickle.load(f)
        self.transform = trans.Compose([trans.Resize(128),
                                        trans.TenCrop(self.image_size),
                                        trans.Lambda(lambda crops:
                                                     torch.stack([trans.ToTensor()(trans.Resize(self.image_size)(crop)) for crop in crops]))])

    def __getitem__(self, index):
        # Read a list of image by index
        target = self.label_list[index:index + 1, :]
        target = torch.from_numpy(target)  # size: (1, 101) 101 classes for ucf101
        # One image example
        file_name = self.data_name_list[index]

        image_index = random.randint(1, self.nFrame_list[index] - self.frames_per_clip)
        img_list = []
        for i in range(self.frames_per_clip):
            img_dir = os.path.join(self.file_dir, file_name, ('frame' + '%06d' % (image_index + i) + '.jpg'))
            img = Image.open(img_dir).convert('RGB')  # convert to RGB
            img_list.append(img)

        # img_tensor = img_tensor[random_crop_index, :]
        img_tensor = torch.cat([self.transform(img_tmp) for img_tmp in img_list], 0)
        target = target.squeeze()

        return img, target  # img size: (frames_per_clip, 10, 3, 112, 112); target size: (101)

    def __len__(self):
        return len(self.data_name_list)


def process_data_from_mat():
    # Read data from mat
    data_dir = '/home/yongyi/ucf101_train/my_code/data'
    name = io.loadmat(os.path.join(data_dir, 'name.mat'))['name']
    label = io.loadmat(os.path.join(data_dir, 'label.mat'))['label']
    one_hot_label = io.loadmat(os.path.join(data_dir, 'one_hot_label.mat'))['one_hot_label']
    data_set = io.loadmat(os.path.join(data_dir, 'set.mat'))['set']
    nFrames = io.loadmat(os.path.join(data_dir, 'nFrames.mat'))['nFrames']

    nObject = name.shape[1]
    train_nObject = np.sum(data_set == 1)
    test_nObject = np.sum(data_set == 2)

    name_list = []
    for i in xrange(nObject):
        a = name[0, i].tolist()
        b = str(a[0])  # example: 'v_ApplyEyeMakeup_g08_c01.avi'
        fname, back = b.split('.')
        name_list.append(fname)  # example: 'v_ApplyEyeMakeup_g08_c01'

    train_name = name_list[:train_nObject]
    test_name = name_list[train_nObject:]

    train_label = one_hot_label[:train_nObject, :]
    test_label = one_hot_label[train_nObject:, :]

    train_nFrames = nFrames[0, :train_nObject]
    test_nFrames = nFrames[0, train_nObject:]

    train_index = label[0, :train_nObject]
    test_index = label[0, train_nObject:]

    f = open(os.path.join(data_dir, 'train_name.pkl'), 'wb')
    pickle.dump(train_name, f)
    f.close()
    f = open(os.path.join(data_dir, 'test_name.pkl'), 'wb')
    pickle.dump(test_name, f)
    f.close()
    f = open(os.path.join(data_dir, 'train_label.pkl'), 'wb')
    pickle.dump(train_label, f)
    f.close()
    f = open(os.path.join(data_dir, 'test_label.pkl'), 'wb')
    pickle.dump(test_label, f)
    f.close()
    f = open(os.path.join(data_dir, 'train_nFrames.pkl'), 'wb')
    pickle.dump(train_nFrames, f)
    f.close()
    f = open(os.path.join(data_dir, 'test_nFrames.pkl'), 'wb')
    pickle.dump(test_nFrames, f)
    f.close()
    f = open(os.path.join(data_dir, 'train_index.pkl'), 'wb')
    pickle.dump(train_index, f)
    f.close()
    f = open(os.path.join(data_dir, 'test_index.pkl'), 'wb')
    pickle.dump(test_index, f)
    f.close()
