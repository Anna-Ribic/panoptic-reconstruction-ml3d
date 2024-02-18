# Increasing Object-Level Reconstruction Quality in Single-Image 3D Scene Reconstruction

## Abstract
While humans can easily infer the 3D structure as well as the complete (panoptic) semantics of a scene from a single image, this task has been a longstanding challenge in the field of computer vision. The task fundamentally prerequisites learning a strong prior of the 3D world. Traditional methods have made significant strides, from generating geometrically coherent structures to learning different
instance semantics. More recent approaches directly learn the 3D panoptic semantics as a whole, yet they fall short in capturing the intricate details and nuances at the object level. This paper introduces a novel approach to bridge this gap by integrating a specialized object-level model into the reconstruction process, thereby leveraging the specialized model’s object-priors. Our approach models panoptic 3D reconstruction as a two-stage problem. We first use the model of Dahnert et al. [Paper](https://manuel-dahnert.com/pdf/dahnert2021panoptic-reconstruction.pdf) to create an initial reconstruction. Then, we leverage the instance masks to extract the object geometries out of the reconstructed scene. We input each of the extracted objectsalong with cropped images from the scene and text labelsinto a diffusion model [5] to refine the rough object-level geometries. Finally, we integrate the refined object geometries back into the initial scene reconstruction to obtain a complete and refined panoptic 3D scene reconstruction.

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
1. Denninger et al. - BlenderProc


