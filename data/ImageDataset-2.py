from io import BytesIO
import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import random
import data.util as Util
import scipy
import scipy.io
import torchvision.transforms as T
import numpy as np
import math
import random
import torch.nn.functional as F


class ImageDataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1):
        self.res = resolution
        self.data_len = data_len
        self.split = split
        self.dataroot = dataroot

        self.path_S0 = Util.get_paths_from_images(dataroot+'/'+'S0')
        self.path_dolp = Util.get_paths_from_images(dataroot+'/'+'dolp')
        self.path_label = Util.get_paths_from_images(dataroot+'/'+'label')
        self.epsilon = torch.tensor(1e-7)

        # if split == 'train':
        #     self.transforms = T.Compose(T.Resize(256, 256), T.RandomHorizontalFlip(), T.ToTensor())
        # else:
        #     self.transforms = T.Compose(T.Resize(256, 256),T.ToTensor())

            
        self.dataset_len = len(self.path_S0)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        random_int      = random.randint(0, 1)
        random_size_int = 2 #random.randint(0, 2)
        random_size = [(256, 320), (384, 480), (512, 640)]

        theta_aug = random.randint(0, 20)


        # img_S0 = Image.open(self.path_S0[index]).convert("L")
        # img_dolp = Image.open(self.path_dolp[index]).convert("L")
        img_label = np.asarray(Image.open(self.path_label[index]))/255#.convert("L")

        img_num= self.path_S0[index].split('.')[0][-4:]

        img_dir = self.dataroot + '/' + 'img/' + img_num + '/'

        img_0 = np.asarray(Image.open(img_dir + '000.png'))/255#.convert("L")
        img_45 = np.asarray(Image.open(img_dir + '045.png'))/255#.convert("L")
        img_90 = np.asarray(Image.open(img_dir + '090.png'))/255#.convert("L")
        img_135 = np.asarray(Image.open(img_dir + '135.png'))/255#.convert("L")

        if random_int:
            img_0 = np.fliplr(img_0)
            img_45 = np.fliplr(img_45)
            img_90 = np.fliplr(img_90)
            img_135 = np.fliplr(img_135)
            img_label = np.fliplr(img_label)

        s0 = 0.5 * (img_0 + img_45 + img_90 + img_135)
        s1 = img_0 - img_90
        s2 = img_45 - img_135

        dolp = np.sqrt(s1 ** 2 + s2 ** 2 + 1e-7) / (s0 + 1e-7)

        img  = np.stack((s0, dolp), axis=0)
        imgc = np.stack((img_0, img_45, img_90, img_135), axis=0)


        if self.split =='train':
            if (theta_aug):
                # s0 = 0.5 * (img_0.numpy() + img_45.numpy() + img_90.numpy() + img_135.numpy())
                # s1 = img_0.numpy() - img_90.numpy()
                # s2 = img_45.numpy() - img_135.numpy()
                #
                # dolp = np.sqrt(s1 ** 2 + s2 ** 2 + 1e-7) / (s0 + 1e-7)

                # normal_dolp = (dolp - dolp.min()) / (dolp.max() - dolp.min())

                psi_rad = 0.5 * np.arctan2(s2, s1)

                psi_deg = np.degrees(psi_rad)

                psi_deg = psi_deg + random.sample(range(0, 91, 5), 1)[0]

                s1_r = dolp * s0 * np.cos(2 * np.radians(psi_deg))
                s2_r = dolp * s0 * np.sin(2 * np.radians(psi_deg))

                img_0_r = 0.5 * (s0 + s1_r * np.cos(2 * np.radians([0])) + s2_r * np.sin(2 * np.radians([0])))
                img_45_r = 0.5 * (s0 + s1_r * np.cos(2 * np.radians([45])) + s2_r * np.sin(2 * np.radians([45])))
                img_90_r = 0.5 * (s0 + s1_r * np.cos(2 * np.radians([90])) + s2_r * np.sin(2 * np.radians([90])))
                img_135_r = 0.5 * (s0 + s1_r * np.cos(2 * np.radians([135])) + s2_r * np.sin(2 * np.radians([135])))

                # img_0_r = torch.from_numpy(img_0_r)
                # img_45_r = torch.from_numpy(img_45_r)
                # img_90_r = torch.from_numpy(img_90_r)
                # img_135_r = torch.from_numpy(img_135_r)

                # imgc = torch.cat((img_0_r, img_45_r, img_90_r, img_135_r), dim=0)
                imgc = np.stack((img_0_r, img_45_r, img_90_r, img_135_r), axis=0)

                # imgc = F.sigmoid(imgc)

        img = torch.from_numpy(img)
        imgc = torch.from_numpy(imgc)
        label = torch.from_numpy(img_label.copy())

            
        return {'img': img, 'imgc': imgc, 'label': label, 'Index': index,  'img_num': int(img_num)}
