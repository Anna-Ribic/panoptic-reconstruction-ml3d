import os
import json
import csv
import h5py
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from termcolor import colored, cprint
from datasets.base_dataset import BaseDataset
from torchvision.transforms.functional import InterpolationMode

class ThreeDFutureMM2ShapeDataset(BaseDataset):

    def initialize(self, opt, phase='train', res=64):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.dataroot = opt.dataroot
        self.phase = opt.phase
        self.res = res

        self.json_file = os.path.join(self.dataroot, f'model_info.json')

        with open(self.json_file, 'r') as f:
            self.data = json.load(f)

        self.model_list = []
        self.text_list = []
        self.img_list = []

        for d in tqdm(self.data, total=len(self.data), desc=f'readinging text data from {self.json_file}'):
            model_id = d['model_id']
            text_info = d['category']
            sdf_path = os.path.join(self.dataroot, f'SDF_v1_64/{model_id}/ori_sample_grid.h5')
            image_path = os.path.join(self.dataroot, f'../../data/3D-FUTURE-mode/{model_id}/image.jpg')

            if not os.path.exists(sdf_path or image_path):
                continue

            self.model_list.append(sdf_path)
            self.text_list.append(text_info)
            # we wrap the image path in a list to maintain consistency with the other dataset classes
            self.img_list.append([image_path])


        self.model_list = self.model_list[:self.max_dataset_size]
        self.text_list = self.text_list[:self.max_dataset_size]
        self.img_list = self.img_list[:self.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.model_list)), 'yellow')

        self.N = len(self.model_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.resize = transforms.Resize((256, 256))

        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        if phase == 'train':
            self.transforms_color = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
            self.transforms = transforms.Compose([
                transforms.RandomAffine(0, scale=(0.7, 1.25), interpolation=InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Normalize(mean, std),
                transforms.Resize((256, 256)),
            ])

        self.transforms_bg = transforms.Compose([
                transforms.RandomCrop(256, pad_if_needed=True, padding_mode='padding_mode'),
                transforms.Normalize(mean, std),
            ])

    def process_img(self, img):
        img_t = self.to_tensor(img)

        _, oh, ow = img_t.shape

        ls = max(oh, ow)

        pad_h1, pad_h2 = (ls - oh) // 2, (ls - oh) - (ls - oh) // 2
        pad_w1, pad_w2 = (ls - ow) // 2, (ls - ow) - (ls - ow) // 2

        img_t = F.pad(img_t[None, ...], (pad_w1, pad_w2, pad_h1, pad_h2), mode='constant', value=0)[0]

        if self.phase == 'train':
            img_fg_mask = (img_t != 0.).float()
            # jitter color first
            img_t = self.transforms_color(img_t)
            img_t_with_mask = torch.cat([img_t, img_fg_mask], dim=0)
            img_t_with_mask = self.transforms(img_t_with_mask)
            img_t, img_fg_mask = img_t_with_mask[:3], img_t_with_mask[3:]
            img_fg_mask = self.resize(img_fg_mask)
            img_t = self.normalize(img_t)
            img_t = self.resize(img_t)
        else:
            img_t = self.transforms(img_t)
        
        return img_t

    def __getitem__(self, index):
        # TODO(f.srambical): add image code
        sdf_h5_file = self.model_list[index]
        text = self.text_list[index]
        image = self.img_list[index]

        with h5py.File(sdf_h5_file, 'r') as f:
            sdf = f['pc_sdf_sample'][:].astype(np.float32)
            sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        imgs = []
        img_paths = []
        imgs_all_view = self.img_list[index]
        # allow replacement. cause in test time, we might only see images from one view
        nimgs = 1
        sample_ixs = np.random.choice(len(imgs_all_view), nimgs)
        for ix in sample_ixs:
            p = imgs_all_view[ix]
            im = Image.open(p).convert('RGB')
            im = self.process_img(im)
            imgs.append(im)
            img_paths.append(p)

        imgs = torch.stack(imgs)
        # img: for one view
        img = imgs[0]
        img_path = img_paths[0]

        ret = {
            'sdf': sdf,
            'img': img,
            'text': text,

            'sdf_path': sdf_h5_file,
            'img_path': img_path,
            'img_paths': img_paths,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'ThreeDFutureMM2ShapeDataset'
