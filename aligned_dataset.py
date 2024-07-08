import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import csv
import torchvision.transforms as transforms
import argparse


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    # assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]



class AlignedDataset(Dataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        # assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = opt.output_nc if opt.direction == 'BtoA' else opt.input_nc
        self.output_nc = opt.input_nc if opt.direction == 'BtoA' else opt.output_nc
        self.file_face_bboxes = os.path.join(opt.dataroot, 'face_bboxes.csv')
        # self.file_face_bboxes = os.path.join('/mnt/liudawei/project/pytorch-CycleGAN-and-pix2pix/datasets/retouchs/face_bboxes.csv')
        self.face_bboxes = {}
        if os.path.exists(self.file_face_bboxes):
            with open(self.file_face_bboxes, encoding='utf-8-sig') as f:
                for row in csv.reader(f, skipinitialspace=True):
                    self.face_bboxes[row[0]] = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]      # xmin, ymin, xmax, ymax

        # self.dir_M = os.path.join(opt.dataroot, 'mask')
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.transform_params_AB = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
        ])
        # self.transform_params_base = transforms.Compose([
        #     transforms.Resize([512, 512]),
        #     transforms.ToTensor(),
        # ])

    def augment_data(self, AB_path):
        AB_dir, AB_name = os.path.split(AB_path)
        AB_name_pre, AB_name_post = os.path.splitext(AB_name)
        AB_name = AB_name_pre.split('_')[0] + AB_name_post
        face_xmin, face_ymin, face_xmax, face_ymax = self.face_bboxes[AB_name]
        face_width, face_height = face_xmax - face_xmin, face_ymax - face_ymin
        face_center = ((face_xmin + face_xmax) / 2, (face_ymin + face_ymax) / 2)
        square_length = face_height if face_height > face_width else face_width
        enlarge_ratio = random.uniform(0.95, 1.2)
        square_length = int(enlarge_ratio * square_length)
        sideScale = 1
        square_bbox = {
            'xmin': int(face_center[0] - sideScale * square_length / 2),
            'xmax': int(face_center[0] + sideScale * square_length / 2),
            'ymin': int(face_center[1] - sideScale * square_length / 2),
            'ymax': int(face_center[1] + sideScale * square_length / 2)
        }
        return square_bbox, AB_name
    
    def get_roi_without_padding(self, img, bbox):
        crop_t = max(bbox['ymin'], 0)
        crop_b = min(bbox['ymax'], img.size[1])
        crop_l = max(bbox['xmin'], 0)
        crop_r = min(bbox['xmax'], img.size[0])
        return img.crop((crop_l, crop_t, crop_r, crop_b))
    
    def randn_rotate_and_center_crop(self, image, target_size):
        angle = np.random.randint(-25, 25)
        image = image.rotate(angle, Image.BICUBIC, expand=1)
        w, h = image.size
        tw, th = target_size, target_size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        image = image.crop((x1, y1, x1 + tw, y1 + th))  # crop()四个参数分别是：(左上角点的x坐标，左上角点的y坐标，右下角点的x坐标，右下角点的y坐标)
        return image

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB_dir, AB_name = os.path.split(AB_path)
        AB_name_pre, AB_name_post = os.path.splitext(AB_name)
        A_name = AB_name_pre.split('_')[0] + AB_name_post

        AB = Image.open(AB_path).convert('RGB')
        
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))
        B = AB.crop((0, 0, w2, h))
        A = AB.crop((w2, 0, w, h))
        # load mask
        # M_path = os.path.join(self.dir_M, A_name)
        # M = Image.open(M_path) 

        # augment by face bbox
        square_bbox, AB_name = self.augment_data(AB_path)
        A = self.get_roi_without_padding(A, square_bbox)
        B = self.get_roi_without_padding(B, square_bbox)
        # M = self.get_roi_without_padding(M, square_bbox)
        # bbox_w, bbox_h = A.size
        # A = self.randn_rotate_and_center_crop(A, bbox_w)
        # B = self.randn_rotate_and_center_crop(B, bbox_h)
        
        # print(AB_path)
        # A_image = util.tensor2im(A)
        # B_image = util.tensor2im(B)
        # A_image.save(os.path.join('/mnt/liudawei/project/pytorch-CycleGAN-and-pix2pix/datasets/retouchs/crop/', AB_name))

        # apply the same transform to both A and B
        A_t = self.transform_params_AB(A)
        B_t = self.transform_params_AB(B)
        # A_Base = self.transform_params_base(A)
        # B_Base = self.transform_params_base(B)

        return {'A': A_t, 'B': B_t, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
    

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epoch', default=300, type=int)
#     parser.add_argument('--dataroot', default='/mnt/liudawei/project/pytorch-CycleGAN-and-pix2pix/datasets/retouchs', type=str, help='train data path')
#     parser.add_argument('--phase', default='train_clean', type=str, help='')
#     parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
#     parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
#     parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
#     parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
#     parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
#     args = parser.parse_args()
#     aligndataset = AlignedDataset(args)


