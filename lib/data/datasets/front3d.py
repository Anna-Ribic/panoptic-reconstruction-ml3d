import os
import random
import zipfile
from pathlib import Path
from typing import Dict, Union, List, Tuple

import numpy as np
import torch.utils.data
from PIL import Image
import pyexr

from lib import data
from lib.data import transforms2d as t2d
from lib.data import transforms3d as t3d
from lib.structures import FieldList
from lib.config import config
from lib.utils.intrinsics import adjust_intrinsic
from lib.structures.frustum import compute_camera2frustum_transform

from scipy.ndimage import rotate
import json
import lib.visualize as vis
import csv

_imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


reverse_label_mapping = {
    -2:-2,
    -1: -1,
    0: 0,    # void
    8: 1,    # cabinet
    71: 1,   # cabinet
    14: 1,   # cabinet
    31: 1,   # cabinet
    36: 1,   # cabinet
    48: 2,   # bed
    58: 2,   # bed
    40: 2,   # bed
    65: 2,   # bed
    27: 2,   # bed
    1: 2,    # bed
    23: 3,   # chair
    28: 3,   # chair
    33: 3,   # chair
    46: 3,   # chair
    47: 3,   # chair
    62: 3,   # chair
    20: 4,   # sofa
    21: 4,   # sofa
    29: 4,   # sofa
    30: 4,   # sofa
    35: 4,   # sofa
    44: 4,   # sofa
    64: 4,   # sofa
    67: 4,   # sofa
    7: 5,    # table
    39: 5,   # table
    49: 5,   # table
    50: 5,   # table
    63: 5,   # table
    56: 6,   # desk
    15: 7,   # dresser
    10: 7,   # dresser
    17: 8,   # lamp
    57: 8,   # lamp
    2: 9,    # other
    3: 9,    # other
    4: 9,    # other
    5: 9,    # other
    6: 9,    # other
    9: 9,    # other
    11: 9,   # other
    12: 9,   # other
    16: 9,   # other
    19: 9,   # other
    24: 9,   # other
    25: 9,   # other
    26: 9,   # other
    34: 9,   # other
    37: 9,   # other
    38: 9,   # other
    41: 9,   # other
    42: 9,   # other
    43: 9,   # other
    51: 9,   # other
    52: 9,   # other
    53: 9,   # other
    54: 9,   # other
    55: 9,   # other
    60: 9,   # other
    61: 9,   # other
    66: 9,   # other
    69: 9,   # other
    70: 9,   # other
    72: 9,   # other
    73: 9,   # other
    13: 10,  # wall
    18: 10,  # wall
    22: 10,  # wall
    32: 10,  # wall
    59: 10,  # wall
    68: 10,  # wall
    45: 11,  # floor
}

def map_labels(label):
    return reverse_label_mapping.get(label, label)


class Front3D(torch.utils.data.Dataset):
    def __init__(self, file_list_path: os.PathLike, dataset_root_path: os.PathLike, fields: List[str],
                 num_samples: int = None, shuffle: bool = False) -> None:
        super().__init__()

        self.dataset_root_path = Path(dataset_root_path)
        self.samples: List = self.load_and_filter_file_list(file_list_path)

        if shuffle:
            random.shuffle(self.samples)

        self.samples = self.samples[:num_samples]

        # Fields defines which data should be loaded
        if fields is None:
            fields = []

        self.fields = fields

        self.image_size = (320, 240)
        self.depth_image_size = (160, 120)
        self.intrinsic = self.prepare_intrinsic()
        self.voxel_size = config.MODEL.PROJECTION.VOXEL_SIZE
        self.depth_min = config.MODEL.PROJECTION.DEPTH_MIN
        self.depth_max = config.MODEL.PROJECTION.DEPTH_MAX
        self.grid_dimensions = config.MODEL.FRUSTUM3D.GRID_DIMENSIONS
        self.truncation = config.MODEL.FRUSTUM3D.TRUNCATION
        self.max_instances = config.MODEL.INSTANCE2D.MAX
        self.num_min_instance_pixels = config.MODEL.INSTANCE2D.MIN_PIXELS
        self.stuff_classes = [0, 10, 11, 12]

        self.frustum_mask: torch.Tensor = self.load_frustum_mask()

        self.transforms: Dict = self.define_transformations()

    def __getitem__(self, index) -> Tuple[str, FieldList]:
        sample_path = self.samples[index]
        scene_id = sample_path.split("/")[0]
        image_id = sample_path.split("/")[1]

        sample = FieldList(self.image_size, mode="xyxy")
        sample.add_field("index", index)
        sample.add_field("name", sample_path)

        #output_path = os.path.join("extracted_meshes_gt/", str(scene_id), str(image_id))
        #os.makedirs(output_path, exist_ok=True)


        try:

            # 2D data
            if "color" in self.fields:
                color = Image.open(self.dataset_root_path / scene_id / f"rgb_{image_id}.png", formats=["PNG"])
                color = self.transforms["color"](color)
                sample.add_field("color", color)

            if "depth" in self.fields:
                depth = pyexr.read(str(self.dataset_root_path / scene_id / f"depth_{image_id}.exr")).squeeze().copy() #[::-1, ::-1].copy()
                depth = depth[:, :, 0]
                depth = self.transforms["depth"](depth)
                sample.add_field("depth", depth) #depth

            if "instance2d" in self.fields:
                segmentation2d = np.load(self.dataset_root_path / scene_id / f"segmap_{image_id}.npz")["data"] #mapped?
                instance_image = segmentation2d[..., 1]
                instance2d = self.transforms["instance2d"](segmentation2d)
                sample.add_field("instance2d", instance2d)

            # 3D data
            needs_weighting = False
            sample.add_field("frustum_mask", self.frustum_mask.clone())

            if "geometry" in self.fields:
                geometry_path = self.dataset_root_path / scene_id / f"geometry_{image_id}.npz"

                # Load Geometry
                with open(geometry_path, 'rb') as file:
                    # Read the dimensions
                    dimX = np.fromfile(file, dtype=np.uint64, count=1)[0]
                    dimY = np.fromfile(file, dtype=np.uint64, count=1)[0]
                    dimZ = np.fromfile(file, dtype=np.uint64, count=1)[0]

                    # Read the data array
                    data = np.fromfile(file, dtype=np.float32)

                    # Reshape the data array based on dimensions
                    data = data.reshape((dimZ, dimY, dimX))

                    diff = torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])

                    data_full = np.pad(np.clip(data, 0, 12), ((int(diff[2]/2), diff[2] - int(diff[2]/2)), (int(diff[1]/2), diff[1] - int(diff[1]/2),), (int(diff[0]/2), diff[0] - int(diff[0]/2))), mode='constant', constant_values=((12, 12),(12,12),(12,12)))
                #geometry = np.load(geometry_path, allow_pickle=True)["data"]
                geometry = data_full

                #Flip and rotate geometry so it aligns with panoptic output
                geometry = np.ascontiguousarray(geometry)#np.flip(geometry, axis=[0, 1]))  # Flip order, thanks for pointing that out.
                geometry = np.copy(np.flip(geometry, axis=[0, 1]))
                geometry = rotate(geometry, 270, axes=(0, 2), reshape=False)
                #Compute weighting
                weighting = np.exp(-(geometry - 3) ** 2) * 5 + 1  # np.ones((256, 256, 256))
                geometry = self.transforms["geometry"](geometry)

                # process hierarchy
                sample.add_field("occupancy_256", self.transforms["occupancy_256"](geometry))
                sample.add_field("occupancy_128", self.transforms["occupancy_128"](geometry))
                sample.add_field("occupancy_64", self.transforms["occupancy_64"](geometry))

                geometry = self.transforms["geometry_truncate"](geometry)
                sample.add_field("geometry", geometry)

                # add frustum mask
                sample.add_field("frustum_mask", self.frustum_mask.clone())
            #Load Semantic and Instance Segmentation
            if "semantic3d" in self.fields or "instance3d" in self.fields:
                segmentation3d_path = self.dataset_root_path / scene_id / f"segmentation_{image_id}.sem" #mapped

                with open(segmentation3d_path, 'rb') as file:
                    # Read the dimensions
                    dimX = np.fromfile(file, dtype=np.uint64, count=1)[0]
                    dimY = np.fromfile(file, dtype=np.uint64, count=1)[0]
                    dimZ = np.fromfile(file, dtype=np.uint64, count=1)[0]

                    # Read the data array
                    num_voxels = np.fromfile(file, dtype=np.uint64, count=1)[0]
                    locations = np.fromfile(file, dtype=np.uint32, count=int(num_voxels * 3)).reshape((int(num_voxels), 3))
                    labels = np.fromfile(file, dtype=np.uint32, count=num_voxels)
                    # Reshape the data array based on dimensions
                semantic_volume = np.zeros((dimX, dimY, dimZ), dtype=np.uint32)

                for i in range(num_voxels):
                    x, y, z = locations[i]
                    semantic_volume[x, y, z] = labels[i]

                semantic_volume = np.swapaxes(semantic_volume, 0, 2)

                diff = torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])

                data_full = np.pad(semantic_volume, (
                (int(diff[2] / 2), diff[2] - int(diff[2] / 2)), (int(diff[1] / 2), diff[1] - int(diff[1] / 2),),
                (int(diff[0] / 2), diff[0] - int(diff[0] / 2))), mode='constant')

                #Flip and Rotate accordingly
                data_full = np.copy(np.flip(data_full, axis=[0, 1]))
                data_full = rotate(data_full, 270, axes=(0, 2), reshape=False)

                #See C** main.cpp
                semantic3d, instance3d = (data_full / 1000), data_full % 1000
                semantic3d, instance3d = semantic3d.astype(np.uint64), instance3d.astype(np.uint8)

                #print(scene_id, image_id,'beforemapping', np.unique(instance3d), np.unique(semantic3d))


                #Map semantics fron Blenerproc labels to Panoptic labels
                mapped_array = np.vectorize(map_labels)(semantic3d)  # TODO
                semantic3d = mapped_array

                #print(scene_id, image_id,'before', np.unique(instance3d), np.unique(semantic3d))

                #Map gt instance labels to Blenerproc labels
                meta_path = self.dataset_root_path / scene_id / "meta.json"
                with open(meta_path, 'r') as file:
                    data_dict = json.load(file)

                mapping_furniture = data_dict['mapping']
                mapping_furniture_reverse = {value: key for key, value in mapping_furniture.items()}

                del mapping_furniture

                csv_path = self.dataset_root_path / scene_id / "class_inst_col_map.csv"
                # Initialize an empty dictionary to store the instanceid -> jid mapping
                jid_dict = {}
                # Open the file and read its contents using the csv.reader
                with open(csv_path, 'r') as file:
                    # Create a CSV reader, assuming ',' as the delimiter
                    csv_reader = csv.reader(file, delimiter=',')

                    # Skip the header line
                    next(csv_reader)

                    # Iterate through each row in the CSV
                    for row in csv_reader:
                        # Extract instanceid and jid from the row
                        instanceid = row[4]
                        jid = row[0]

                        # Populate the dictionary
                        jid_dict[instanceid] = jid

                mapping_3d = {}
                for id in np.unique(instance3d):
                    if id == 0:
                        continue
                    if id in mapping_furniture_reverse.keys():
                        f_id = mapping_furniture_reverse[id]
                        if f_id in jid_dict.keys():
                            mapping_3d[int(id)] = int(jid_dict[mapping_furniture_reverse[id]])

                if "semantic3d" in self.fields:
                    semantic3d = self.transforms["semantic3d"](semantic3d)
                    sample.add_field("semantic3d", semantic3d)

                    # process semantic3d hierarchy
                    sample.add_field("semantic3d_64", self.transforms["segmentation3d_64"](semantic3d))
                    sample.add_field("semantic3d_128", self.transforms["segmentation3d_128"](semantic3d))

                if "instance3d" in self.fields:
                    # Ensure consistent instance id shuffle between 2D and 3D instances
                    instance_mapping = sample.get_field("instance2d").get_field("instance_mapping")
                    transform = t3d.Compose([t3d.ToTensor(dtype=torch.long), t3d.Mapping(mapping=mapping_3d, ignore_values=[0]),t3d.Mapping(mapping=instance_mapping, ignore_values=[0]) ])
                    instance3d = transform(instance3d)#mapping TODO
                    #instance3d = self.transforms["instance3d"](instance3d, mapping=instance_mapping)#mapping TODO
                    sample.add_field("instance3d", instance3d)
                    # process instance3d hierarchy
                    sample.add_field("instance3d_64", self.transforms["segmentation3d_64"](instance3d))
                    sample.add_field("instance3d_128", self.transforms["segmentation3d_128"](instance3d))

                print(scene_id, image_id,'after', torch.unique(instance3d), torch.unique(semantic3d))

                furniture = torch.unique(semantic3d)
                furniture = furniture[(furniture !=0) & (furniture !=10) & (furniture !=11)]
                if len(furniture)==0:
                    print('Ignore: '+ str(scene_id)+'/'+str(image_id))
                    with open('ignore.txt', 'a') as ig:
                        ig.write(str(scene_id)+'/'+str(image_id)+'\n')


            #weighting = np.ones((256, 256,256))#TODO remove
            weighting = self.transforms["weighting3d"](weighting)
            sample.add_field("weighting3d", weighting)

            # Process weighting mask hierarchy
            sample.add_field("weighting3d_64", self.transforms["weighting3d_64"](weighting))
            sample.add_field("weighting3d_128", self.transforms["weighting3d_128"](weighting))

            """camera2frustum = compute_camera2frustum_transform(depth.intrinsic_matrix.cpu(),
                                                              torch.tensor(color.size()) / 2.0,
                                                              config.MODEL.PROJECTION.DEPTH_MIN,
                                                              config.MODEL.PROJECTION.DEPTH_MAX,
                                                              config.MODEL.PROJECTION.VOXEL_SIZE)

            camera2frustum[:3, 3] += (torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])) / 2
            frustum2camera = torch.inverse(camera2frustum)
            print(frustum2camera)

            vis.write_distance_field(geometry.squeeze(), None, output_path + "/mesh_geometry.ply", transform=frustum2camera)
            vis.write_distance_field(geometry.squeeze(), semantic3d.squeeze(), output_path + "/mesh_semantics.ply", transform=frustum2camera)
            vis.write_distance_field(geometry.squeeze(), instance3d.squeeze(), output_path + "/mesh_instances_test.ply", transform=frustum2camera)

            vis.write_distance_field(geometry.squeeze(), instance3d.squeeze(), output_path + "/mesh_instances.ply",
                                     semantic_labels=semantic3d.squeeze(), transform=frustum2camera)"""




            return sample_path, sample
        except Exception as e:
            print(sample_path)
            print(e)
            return None, sample

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def load_and_filter_file_list(file_list_path: os.PathLike) -> List[str]:
        with open(file_list_path) as f:
            content = f.readlines()

        images = [line.strip() for line in content]

        return images

    def load_frustum_mask(self) -> torch.Tensor:
        mask_path = self.dataset_root_path / "frustum_mask.npz"
        mask = np.load(str(mask_path))["mask"]
        mask = torch.from_numpy(mask).bool()

        return mask

    def define_transformations(self) -> Dict:
        transforms = dict()

        # 2D transforms
        transforms["color"] = t2d.Compose([
            t2d.ToTensor(),
            t2d.Normalize(_imagenet_stats["mean"], _imagenet_stats["std"])
        ])

        transforms["depth"] = t2d.Compose([
            t2d.ToImage(),
            t2d.Resize(self.depth_image_size, Image.NEAREST),
            t2d.ToNumpyArray(),
            t2d.ToTensorFromNumpy(),
            t2d.ToDepthMap(self.intrinsic)  # 3D-Front has single intrinsic matrix
        ])

        transforms["instance2d"] = t2d.Compose([
            t2d.SegmentationToMasks(self.image_size, 50, self.max_instances, True, self.stuff_classes)#self.num_min_instance_pixels
        ])

        # 3D transforms
        transforms["geometry"] = t3d.Compose([
            t3d.ToTensor(dtype=torch.float),
            t3d.Unsqueeze(0),
            t3d.ToTDF(truncation=12)
        ])

        transforms["geometry_truncate"] = t3d.ToTDF(truncation=self.truncation)

        transforms["occupancy_64"] = t3d.Compose([t3d.ResizeTrilinear(0.25), t3d.ToBinaryMask(8), t3d.ToTensor(dtype=torch.float)])
        transforms["occupancy_128"] = t3d.Compose([t3d.ResizeTrilinear(0.5), t3d.ToBinaryMask(6), t3d.ToTensor(dtype=torch.float)])
        transforms["occupancy_256"] = t3d.Compose([t3d.ToBinaryMask(self.truncation), t3d.ToTensor(dtype=torch.float)])

        transforms["weighting3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.float), t3d.Unsqueeze(0)])
        transforms["weighting3d_64"] = t3d.ResizeTrilinear(0.25)
        transforms["weighting3d_128"] = t3d.ResizeTrilinear(0.5)

        transforms["semantic3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.long)])

        transforms["instance3d"] = t3d.Compose([t3d.ToTensor(dtype=torch.long), t3d.Mapping(mapping={}, ignore_values=[0])])

        transforms["segmentation3d_64"] = t3d.Compose([t3d.ResizeMax(8, 4, 2)])
        transforms["segmentation3d_128"] = t3d.Compose([t3d.ResizeMax(4, 2, 1)])

        return transforms

    def prepare_intrinsic(self) -> torch.Tensor:
        intrinsic = np.array(config.MODEL.PROJECTION.INTRINSIC).reshape((4, 4))
        intrinsic_adjusted = adjust_intrinsic(intrinsic, self.image_size, self.depth_image_size)
        intrinsic_adjusted = torch.from_numpy(intrinsic_adjusted).float()

        return intrinsic_adjusted
