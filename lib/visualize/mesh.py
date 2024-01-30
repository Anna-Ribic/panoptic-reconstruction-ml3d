import os
from typing import Union, Tuple, Optional

import numpy as np
import torch

import marching_cubes as mc

from lib.utils.transform import coords_multiplication
from . import io, utils

semantics = {
        (0, 0, 0) : 'None',
        (174, 199, 232) : 'Wall',		# wall
        (152, 223, 138) : 'Floor',		# floor
        (31, 119, 180) : 'Cabinet', 		# cabinet
        (255, 187, 120) : 'Bed',		# bed
        (188, 189, 34): 'Chair', 		# chair
        (140, 86, 75) : 'Sofa',  		# sofa
        (255, 152, 150) : 'Table',		# table
        (214, 39, 40):  'Door',  		# door
        (197, 176, 213): 'Window',		# window
        (148, 103, 189): 'Bookshelf',		# bookshelf
        (196, 156, 148): 'Pictue',		# picture
        (23, 190, 207): 'Counter', 		# counter
        (247, 182, 210): 'Desk',		# desk
        (219, 219, 141): 'Curtain',		# curtain
        (255, 127, 14): 'Fridge', 		# refrigerator
        (158, 218, 229): 'ShowerCurtain',		# shower curtain
        (44, 160, 44): 'Toilet',  		# toilet
        (112, 128, 144): 'Sink',		# sink
        (227, 119, 194): 'Bathtub',		# bathtub
        (82, 84, 163): 'Other' 		# otherfurn
}
"""
sem_labels = [
        'None',
        'Wall',		# wall
        'Floor',		# floor
        'Cabinet', 		# cabinet
        'Bed',		# bed
        'Chair', 		# chair
        'Sofa',  		# sofa
        'Table',		# table
        'Door',  		# door
        'Window',		# window
        'Bookshelf',		# bookshelf
        'Picture',		# picture
        'Counter', 		# counter
        'Unknown1',
        'Desk',		# desk
        'Unknown2',
        'Curtain',		# curtain
        'Unknown3',
        'Unknown4',
        'Unknown5',
        'Unknown6',
        'Unknown7',
        'Unknown8',
        'Unknown9',
        'Fridge', 		# refrigerator
        'Unknown10',
        'Unknown11',
        'Unknown12',
        'ShowerCurtain',		# shower curtain
    'Unknown13',
    'Unknown14',
    'Unknown15',
    'Unknown16',
    'Toilet',  		# toilet
        'Sink',		# sink
    'Unknown17',
    'Bathtub',		# bathtub
    'Unknown18',
    'Unknown19',
        'Other',  		# otherfurn
    'Unknown20',
    'Unknown21',
    ]"""

sem_labels = [
    'None',
    'Cabinet',
    'Bed',
    'Chair',
    'Sofa',
    'Table',
    'Desk',
    'Dresser',
    'Lamp',
    'Other',
    'Wall',
    'Floor'
]


def write_distance_field(distance_field: Union[np.array, torch.Tensor], labels: Optional[Union[np.array, torch.Tensor]],
                         output_file: os.PathLike, iso_value: float = 1.0, truncation: float = 3.0,
                         color_palette=None, transform=None, semantic_labels=None) -> None:
    if isinstance(distance_field, torch.Tensor):
        distance_field = distance_field.detach().cpu().numpy()

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if labels is not None:
        if semantic_labels is not None:
            semantic_labels = semantic_labels.detach().cpu().numpy()
            print(str(len(np.unique(labels)))+' instances detected.')
            for v in np.unique(labels):
                distance_field_temp = distance_field.copy()
                distance_field_temp[labels!=v] = truncation
                sem = np.unique(semantic_labels[labels==v])[0]
                sem_name= sem_labels[sem]
                print(f'Instance {v+1} has label {sem_name}', )
                vertices, colors, triangles = get_mesh_with_semantics(distance_field_temp, labels, iso_value, truncation, color_palette)

                if transform is not None:
                    if isinstance(transform, torch.Tensor):
                        transform = transform.detach().cpu().numpy()

                vertices = coords_multiplication(transform, vertices)
                io.write_ply(vertices, colors, triangles, str(output_file)[:-4]+str(v+1)+sem_name+'.ply')

            
        vertices, colors, triangles = get_mesh_with_semantics(distance_field, labels, iso_value, truncation,
                                                              color_palette)
    else:
        vertices, triangles = get_mesh(distance_field, iso_value, truncation)
        colors = None

    if transform is not None:
        if isinstance(transform, torch.Tensor):
            transform = transform.detach().cpu().numpy()

        vertices = coords_multiplication(transform, vertices)

    io.write_ply(vertices, colors, triangles, output_file)


def get_mesh(distance_field: np.array, iso_value: float = 1.0, truncation: float = 3.0) -> Tuple[np.array, np.array]:
    vertices, triangles = mc.marching_cubes(distance_field, iso_value, truncation)
    return vertices, triangles


def get_mesh_with_semantics(distance_field: np.array, labels: np.array, iso_value: float = 1.0, truncation: float = 3.0,
                            color_palette=None) -> Tuple[np.array, np.array, np.array]:
    labels = labels.astype(np.uint32)
    color_volume = utils.lookup_colors(labels, color_palette)
    vertices, colors, triangles = get_mesh_with_colors(distance_field, color_volume, iso_value, truncation)

    return vertices, colors, triangles


def get_mesh_with_colors(distance_field: np.array, colors: np.array, iso_value: float = 1.0,
                         truncation: float = 3.0) -> Tuple[np.array, np.array, np.array]:
    vertices, triangles = mc.marching_cubes_color(distance_field, colors, iso_value, truncation)
    colors = vertices[..., 3:]
    vertices = vertices[..., :3]

    return vertices, colors, triangles
