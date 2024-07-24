import os
import torch
import torchvision
import random
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in ['.mat'])


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)

def get_paths_from_mat(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_mat_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = T.Compose(
            [ T.ToTensor()])
to2tensor = T.Compose(
            [T.ToTensor()]) #T.Resize((512, 640)),
hflip = torchvision.transforms.RandomHorizontalFlip()
rcrop = torchvision.transforms.RandomCrop(size=256)
resize = torchvision.transforms.Resize(size=256)

# augmentations for images
def transform_augment(img, split='train', random_int=0, random_size=(512,640)):

    to_tensor = T.Compose([T.Resize(random_size),T.ToTensor()])

    if split == 'train':

       img = to_tensor(img)
       if random_int==1:
           img = F.hflip(img)

    else:

       img = to2tensor(img)
    # if split == 'train':
    #     if img.size(1) < res:
    #         img = resize(img)
    #     elif img.size(1) > res:
    #         img = rcrop(img)
    #     else:
    #         img=img
    #     img = hflip(img)
    # ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return img


def transform_label(img, split='train', size=(512,640), random_int=0):

    # img = img.resize((512,512))
    if split == 'train':
       img = img.resize(size)
       if random_int==1:
           img = img.transpose(Image.FLIP_LEFT_RIGHT)

    img = torch.from_numpy(np.asarray(img)/255)

    return img

def transform_augment_cd(img, split='val', min_max=(0, 1)):
    img = totensor(img)
    ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return ret_img
