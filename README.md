# Increasing Object-Level Reconstruction Quality in Single-Image 3D Scene Reconstruction

## Abstract
 Panoptic 3D Scene Reconstruction describes the joint task of geometric reconstruction, 3D semantic segmentation, and 3D instance
segmentation. A multitude of tasks in Robotics, Augmented Reality and Human-Computer Interaction rely on this comprehensive understanding of 3d scenes. Building upon the method introduced by Dahnert et al. [3], which performs panoptic 3D scene reconstruction from a single RGB image, our proposal aims to enhance the visual clarity and discernibility of the generated geometry through a Retrieval-inspired approach.  Leveraging a 3D asset generation framework [4] , we conduct object-level reconstruction conditioned on semantic labels and image input, further advancing the capabilities of panoptic 3D scene reconstruction.


## Environment
The code was tested with the following configuration:
- Ubuntu 20.04
- Python 3.8
- Pytorch 1.7.1
- CUDA 10.2
- Minkowski Engine 0.5.1, fork
- Mask RCNN Benchmark
- Nvidia 2080 Ti, 11GB

## Installation
```
# Basic conda enviromnent: Creates new conda environment `panoptic`
conda env create --file environment.yaml
conda activate panoptic
```

### MaskRCNN Benchmark
Follow the official instructions to install the [maskrcnn-benchmark repo](https://github.com/facebookresearch/maskrcnn-benchmark).

### Minkowski Engine (fork, custom)
Follow the instructions to compile [our forked Minkowski Engine version](https://github.com/xheon/MinkowskiEngine) from source.

### Compute library
Finally, compile this library. 

```
# Install library
cd lib/csrc/
python setup.py install
```

## Inference
TODO


## Datasets

### 3D-FRONT [1]

The [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) indoor datasets consists of 6,813 furnished apartments.  
We use Blender-Proc [2] to render photo-realistic images from individual rooms.
We use version from 2020-06-14 of the data.

<p align="center">
    <img width="100%" src="images/front3d_samples.jpg"/>
</p>

#### Download:
TDOD

#### Modifications:
TODO

# References

1. Fu et al. - 3d-Front: 3d Furnished Rooms with Layouts and Semantics
2. Denninger et al. - BlenderProc
3. TODO
4. TODO


