import os
import json
import csv
import h5py
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from datasets.base_dataset import BaseDataset

class PanopticMM2ShapeDataset(BaseDataset):

    def initialize(self, opt):
        self.opt = opt
        self.max_dataset_size = opt.max_dataset_size
        self.dataroot = opt.dataroot
        self.phase = opt.phase
        self.res = opt.res

        self.text_csv = os.path.join(self.dataroot, f'test.csv')

        with open(self.text_csv) as f:
            reader = csv.reader(f, delimiter=',')
            self.header = next(reader, None)
            self.data = [row for row in reader]

        self.model_list = []
        self.text_list = []
        self.gt_sdf_list = []

        for d in self.data:
            model_id, text, gt_sdf_path = d

            sdf_path = os.path.join(self.dataroot, f'SDF/{model_id}.h5')

            if not os.path.exists(sdf_path) or not os.path.exists(gt_sdf_path):
                continue

            self.model_list.append(sdf_path)
            self.text_list.append(text)
            self.gt_sdf_list.append(gt_sdf_path)

        self.N = min(len(self.model_list), self.max_dataset_size)

    def __getitem__(self, index):
        # TODO(f.srambical): add image code
        model_id = self.model_list[index]
        text = self.text_list[index]
        gt_sdf_path = self.gt_sdf_list[index]

        with h5py.File(model_id, 'r') as f:
            sdf = f['sdf'][:].astype(np.float32)
            sdf = torch.Tensor(sdf).view(1, self.res, self.res, self.res)

        with h5py.File(gt_sdf_path, 'r') as f:
            gt_sdf = f['sdf'][:].astype(np.float32)
            gt_sdf = torch.Tensor(gt_sdf).view(1, self.res, self.res, self.res)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)
            gt_sdf = torch.clamp(gt_sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            'img': None,
            'text': text,
            'gt_sdf': gt_sdf,
            'sdf_path': model_id,
            'gt_sdf_path': gt_sdf_path
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'PanopticMM2ShapeDataset'
