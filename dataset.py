import torch.utils.data as data
import torch
import numpy as np
import random
import cv2
from glob import glob
import os


class LoadData(data.Dataset):
    def __init__(self, img_path, img_name, mask_name):
        super(LoadData, self).__init__()

        self.data_path = img_path
        self.img_name = img_name
        self.mask_name = mask_name

        self.imgpath = []
        self.imgs = []
        self.maskpath = []
        self.masks = []

        fol1 = glob(os.path.join(self.data_path, '*'))

        for i in fol1:
            n = i + self.img_name
            x = np.sort(glob(n))
            self.imgpath.extend(x)

        for i in self.imgpath:
            img = cv2.imread(i, 0)
            img_tensor = torch.from_numpy(img/255.).float()
            self.imgs.append(img_tensor)

        fol2 = glob(os.path.join(self.data_path, '*'))

        for i in fol2:
            n = i + self.mask_name
            x = np.sort(glob(n))
            self.maskpath.extend(x)

        for i in self.maskpath:
            img = cv2.imread(i, 0)
            img_tensor = torch.from_numpy(img/255.).float()
            self.masks.append(img_tensor)

        # with open('name.txt', 'w') as f:
        #     for i in range(len(self.imgpath)):
        #         f.write(self.imgpath[i] + '\n')
    
    def __getitem__(self, index):

        tensor_img = self.imgs
        tensor_mask = self.masks
        imgpath = self.imgpath
        maskpath = self.maskpath

        # print(len(tensor_img))
        # print(len(tensor_mask))

        return tensor_img[index], tensor_mask[index], imgpath[index], maskpath[index]       


    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    a = LoadData('/home/comp/e9251130/code/COMP4026_project/Infection Segmentation Data/Train', '/images/*', '/infection masks/*')
    lungs, mask = a.__getitem__(0)

    print(lungs)
    print(lungs.shape)
    print(mask.shape)

    print(a.__len__())