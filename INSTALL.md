# Installation
ComA has been developed and tested on Ubuntu 20.04 with an NVIDIA GeForce RTX 3090 GPU device. To get started, follow the installation instructions below.

## Environment Setup

```shell
conda create -n coma python=3.9
conda activate coma

# install PyTorch
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.6.1" libcusolver-dev

# install detectron2
git clone https://github.com/NVlabs/ODISE.git
cd ODISE
mkdir third_party/Mask2Former/build/
mkdir third_party/Mask2Former/build/temp.linux-x86_64-cpython-39
pip install -e .
cd ..

# install PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu116_pyt1131/download.html

# install visualizers
pip install blenderproc
pip install mayavi

# install other dependencies
pip install diffusers==0.20.2 accelerate safetensors transformers
pip install numpy==1.23.1 loguru numba filterpy flatten-dict smplx trimesh==3.23.5 jpeg4py chumpy easydict pickle5 torchgeometry networkx==2.8 pysdf mayavi PyQt5==5.14.2 jupyter yq tqdm supervision Pillow==9.5.0 open3d plyfile openai configer
pip install pyopengl==3.1.4 pyrender==0.1.45
pip install segment-anything
pip install -e .
```


## Environment Setup for Blender Python

ComA uses Blender as the main visualizer. To integrate various features in Blender (e.g., rendering) with a machine learning framework, we install PyTorch on top of Blender python.

```shell
blenderproc pip install tqdm # this will automatically install Blender on your machine
[Blender pip path] install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
export PATH="[Blender executable directory path]:$PATH"
```

Typically, the installed [Blender pip path] will be in a form like `~/blender/blender-3.5.1-linux-x64/3.5/python/bin/pip3`, and the Blender executable directory path will be in a form like `/home/[USER NAME]/blender/blender-3.5.1-linux-x64`. Check the paths and customize for your setting.


## Preparing Pre-trained Weights

ComA leverages off-the-shelf segmentation model [PointRend](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend) for Adaptive Mask Inpainting, and off-the-shelf 3D Human Estimator [Hand4Whole](https://github.com/mks0601/Hand4Whole_RELEASE) for uplifting 2D HOI Images to 3D.

### Setup PointRend

Download the pre-trained PointRend Instance Segmentation model from the [github](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend) with R50-FPN backbone and 3x lr schedule. Place the model in below the path `imports/pointrend/weights/`. The name of the model is likely `model_final_edd263.pkl`. The final directory should be as:

```
imports
└── pointrend
    ├── config 
    │   ├── Base-PointRend-RCNN-FPN.yaml
    │   ├── Base-RCNN-FPN.yaml
    │   └── pointrend_rcnn_R_50_FPN_3x_coco.yaml
    └── weights
        └── model_final_edd263.pkl
```

### Setup Hand4Whole

Download the pre-trained Hand4Whole model from the [github](https://github.com/mks0601/Hand4Whole_RELEASE?tab=readme-ov-file#models) trained on H36M+MPII+MSCOCO. Place the model below the path `imports/hand4whole/`. The name of the model is likely `snapshot_6.pth.tar`. The final directory should be as:

```
imports
└── hand4whole
    ├── common 
    ├── main 
    ├── LICENSE 
    ├── README.md 
    └── snapshot_6.pth.tar
```

To setup the human model files, download the files from [[smpl]](https://smpl.is.tue.mpg.de/) [[smplx]](https://smpl-x.is.tue.mpg.de/) [[SMPLX_to_J14.pkl]](https://github.com/vchoutas/expose#preparing-the-data) [[mano]](https://mano.is.tue.mpg.de/) [[flame]](https://flame.is.tue.mpg.de/). Place the files below the path `imports/hand4whole/common/human_model_files/`. The final directory should be as:

```
imports
└── hand4whole
    └── common 
        └── human_model_files
            ├── smpl 
            │   └── SMPL_NEUTRAL.pkl
            ├── smplx
            │   ├── MANO_SMPLX_vertex_ids.pkl
            │   ├── SMPL-X__FLAME_vertex_ids.npy
            │   ├── SMPLX_NEUTRAL.pkl
            │   └── SMPLX_to_J14.pkl
            ├── mano
            │   ├── MANO_LEFT.pkl
            │   └── MANO_RIGHT.pkl
            └── flame
                ├── flame_dynamic_embedding.npy
                ├── flame_static_embedding.pkl
                └── FLAME_NEUTRAL.pkl
```

## Setup Dataset

We use existing 3D object datasets (e.g., [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future), [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/), [InterCap](https://intercap.is.tue.mpg.de/index.html), [SAPIEN](https://sapien.ucsd.edu/), [ShapeNet](https://shapenet.org/)) for learning ComA along with [our collected SketchFab data](https://github.com/snuvclab/coma?tab=readme-ov-file#download-the-dataset). You may need some manual canonicalization (orientation, position, scale) for BEHAVE, InterCap, and SAPIEN as our multiview camera is installed toward to the area close to the origin. You can use 3D-FUTURE, ShapeNet, and our collected SketchFab data directly without processing. 

Because of the license issue, we cannot share our processed meshes for BEHAVE and InterCap. We provide canonicalization code for certain objects (BEHAVE: backpack, InterCap: suitcase) in `utils/canonicalize.py`. For others, try to canonicalize it manually and  modifiy rendering configurations in `constants/generation/assets.py`. Use following commands for canonicalizing BEHAVE: backpack, and InterCap: suitcase.

```shell
blenderproc debug utils/canonicalize.py --supercategory "BEHAVE" --category "backpack" --obj_path [obj_path]
blenderproc debug utils/canonicalize.py --supercategory "INTERCAP" --category "suitcase" --obj_path [obj_path]
```

For SAPIEN, which is licensed under the MIT License, we provide our processed mesh data via [Google Drive](https://drive.google.com/file/d/1GBLPm_XRTkooaof3GHN1oSt2zosYbHWr/view?usp=sharing). Place the folder under `data/`. The final directory should be as:

```
data
├── 3D-FUTURE-model
├── BEHAVE
├── INTERCAP
├── SAPIEN
├── ShapeNetCore.v2
└── SketchFab
```

Note that you don’t need all of the object mesh data. Only prepare the object data that you want to use for learning ComA.

## Optional Setup

### (Optional) Setup VPoser

To run our application, the optimization framework, download the pre-trained weight for VPoser from the [website](https://smpl-x.is.tue.mpg.de/). Place the weight under the `imports/vposer/snapshots/`. The name of the model is likely `TR00_E096.pt`. The final directory should be as:

```
imports
└── vposer
    ├── snapshots 
    │   └── TR00_E096.pt
    ├── model_loader.py 
    ├── prior.py
    ├── TR00_004_00_WO_accad.ini
    └── vposer_smpl.py
```

### (Optional) Setup Point Cloud Visualizer on Blender

To use Blender for Point Cloud Visualization, download the [addon](https://github.com/uhlik/bpy/blob/master/space_view3d_point_cloud_visualizer.py) file and place it at the `addons` folder of the Blender. The final directory should be similar as: 

```
blender
└──blender-3.5.1-linux-x64
    └── 3.5
        └── scripts
            ├── ...
            └── space_view3d_point_cloud_visualizer.py
```