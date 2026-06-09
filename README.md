# RoadSurf: Learning Implicit Road Surface Fields for Roadside Monocular 3D Object Detection

## Introduction

RoadSurf is a roadside monocular 3D object detection framework for infrastructure cameras. Roadside cameras are usually mounted at large heights with strong pitch angles, and real road scenes often contain ramps, medians, sidewalks, and multi-level structures. These conditions make the common single-plane road assumption unreliable and can introduce depth bias for objects on non-flat roads. RoadSurf addresses this problem by replacing the rigid plane prior with an implicit road surface field in BEV coordinates. It projects image features to a BEV grid using the calibrated plane, learns a residual 2.5D height field, and jointly trains this field with a depth-aware monocular transformer detector. Road-surface losses couple the learned surface with object geometry, including pointwise supervision from object bases, smoothness and plane regularization, and consistency between predicted boxes, the calibrated plane, and the learned surface.

## Installation

1. Clone this project and create a conda environment:

    ```bash
    git clone https://github.com/lixiaole2016/RoadSurf-OPEN
    cd RoadSurf

    conda create -n RoadSurf python=3.9
    conda activate RoadSurf
    ```

2. Install PyTorch and torchvision matching your CUDA version. For example:

    ```bash
    conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

3. Install requirements and compile deformable attention:

    ```bash
    pip install -r requirements.txt

    cd lib/models/monodetr/ops/
    bash make.sh
    cd ../../../..
    ```

## Data Preparation

Prepare Rope3D in a KITTI-style directory structure:

```text
RoadSurf/
├── data/
│   └── Rope3D/
│       ├── ImageSets/
│       │   ├── train.txt
│       │   ├── val.txt
│       │   └── test.txt
│       ├── training/
│       │   ├── image_2/
│       │   ├── calib/
│       │   ├── label_2/
│       │   └── denorm/
│       └── testing/
│           ├── image_2/
│           ├── calib/
│           ├── label_2/
│           └── denorm/
```


## Get Started

### Train

Modify the dataset path and GPU settings in `configs/roadsurf_rope3d.yaml`, then run:

```bash
bash train.sh configs/roadsurf_rope3d.yaml
```

### Test

The best checkpoint will be evaluated by default:

```bash
bash test.sh configs/roadsurf_rope3d.yaml
```

## Acknowledgement

This project is not possible without the following codebases.
[MonoDETR](https://github.com/ZrrSkywalker/MonoDETR), 
[Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR), 
and [MonoDLE](https://github.com/xinzhuma/monodle).

