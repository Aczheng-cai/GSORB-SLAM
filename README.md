<p align="center">

<h2 align="center">GSORB-SLAM: Gaussian Splatting SLAM benefits from<br>ORB features and Transmittance information</h2>

<h3 align="center">
    <a href="https://ieeexplore.ieee.org/document/11091447"> <img src="https://img.shields.io/badge/IEEE-RA--L-004c99"> </a>
    <a href="https://arxiv.org/abs/2410.11356" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2410.21955-blue?logo=arxiv&color=%23B31B1B" alt="Paper arXiv"></a>
    <a href="https://aczheng-cai.github.io/gsorb-slam.github.io/" target="_blank">
    <img src="https://img.shields.io/badge/Project-Page-a" alt="Project Page"></a>
    <a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</h3>
<div align="center"></div>

<div align=center> <img src="https://github.com/Aczheng-cai/GSORB-SLAM/blob/main/pipline.png" width="850"/> </div>

## 💡 News

* **[14 July 2025]** 🎉 Our paper **GSORB-SLAM** has been accepted to **IEEE RA-L 2025**!
* 🌟 If you find this project interesting, please consider starring it to support us!

## 🛠️ Installation

Our project has been tested on Ubuntu 20.04 and 22.04 with CUDA 11.8.

Clone the repository and create the conda environment:

```
git clone --recurse https://github.com/Aczheng-cai/GSORB-SLAM.git GSORB_SLAM
conda create -n gsorbslam python=3.10
conda activate gsorbslam
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

# You need to configure the Conda environment in the CMakeLists file.
set(ENVIRONMENT_DIR your_anconda3_path/envs/gsorbslam)
```

### Libtorch (C++)

```
# In a Terminal
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcu118.zip -O libtorch-cu118.zip
unzip libtorch-cu118.zip -d ~/GSORB_SLAM/Thirdparty/
rm libtorch-cu118.zip
```

### OpenCV with opencv\_contrib and CUDA

Our implementation has been tested with [OpenCV 4.5.5](https://github.com/opencv/opencv/archive/refs/tags/4.5.5.tar.gz) and [OpenCV 4.9.0](https://github.com/opencv/opencv/archive/refs/tags/4.9.0.tar.gz). In this document, we take [OpenCV 4.9.0](https://github.com/opencv/opencv/archive/refs/tags/4.9.0.tar.gz) as the default example.

Find the corresponding versions in both the [OpenCV realeases](https://github.com/opencv/opencv/releases) and[ opencv\_contrib]([opencv\_contrib](https://github.com/opencv/opencv_contrib/tags)) repositories (e.g., [OpenCV 4.9.0](https://github.com/opencv/opencv/archive/refs/tags/4.9.0.tar.gz) and [opencv\_contrib 4.9.0](https://github.com/opencv/opencv_contrib/archive/refs/tags/4.9.0.tar.gz), download them into the same directory (e.g., `~/opencv`), and extract them. Then, open a terminal and run:

```
cd ~/opencv
cd opencv-4.9.0/
mkdir build && cd build

# The build options we used in our tests:
cmake -DCMAKE_BUILD_TYPE=RELEASE -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON -DWITH_NVCUVID=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 -DOPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-4.9.0/modules" -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_JASPER=ON -DBUILD_CCALIB=ON -DBUILD_JPEG=ON -DWITH_FFMPEG=ON ..

make -j$(nproc)
sudo make install
```

### Eigen3

Required by g2o (see below). Download and install instructions can be found at: [http://eigen.tuxfamily.org](http://eigen.tuxfamily.org/).

### Thirdparty

Install third-party libraries, including Gaussian Splatting, tinyply, yaml, and wandb (more may be added later ⏳).

```
#install tinyply
cd ~/GSORB_SLAM/Thirdparty/diff_gaussian_rasterization/
conda activate gsorbslam
python setup.py install

#install tinyply
cd ../tinyply
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

#install yaml
cd ../../yaml-cpp
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

#install pangolin-0.6. We recommend using version 0.6.
cd ../../
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git Pangolin
cd Pangolin
git checkout v0.6
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

## Installation of GSORB-SLAM🎉️

```
cd ~/GSORB_SLAM
chmod +x ./build.sh
sh build.sh
```

## 📈 Evaluation dependencies（Option）

This step can be skipped if you don't intend to run evaluation directly in C++ after execution.  **Note:** You need to set `enable` under the `Evaluation` section in the YAML file to `false`. 💡Don't worry — we also provide a Python version of the offline evaluation (replay.py).

```
# Download the LPIPS model and convert the .pth file to .pt.
cd ~/GSORB_SLAM/scripts/
python gen_eval_model.py

# Note
Evalution:
  enable: true   <----If you do not wish to evaluate the results directly, please modify this setting.
```

## 🧾Datasets

### TUM-RGBD

```
bash srcipts/dataset_utils/download_tum.sh
```

### Replica

```
bash scripts/dataset_utils/download_replica.sh
```

### Scannet

Please follow the data downloading procedure on the [ScanNet](http://www.scan-net.org/) website, and extract color/depth frames from the `.sens` file using this [code](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).

```
datasets
├── Scannet
│   ├── scene0000_00
│   │   ├── cameras.json
│   │   ├── color
│   │   ├── depth
│   │   ├── intrinsic
│   │   ├── pose
│   │   └── groundtruth.txt  <----bash pose2traj.sh
├── Replica
└── TUM_RGBD
```

1. ScanNet requires using the script `pose2traj.sh` to convert poses into 4×4 continue trajectory matrices for evaluation.
   
   ```
   bash scripts/dataset_utils/pose2traj.sh
   ```


## 🚀 Run

Run the TUM RGB-D dataset sequence `rgbd_dataset_freiburg1_desk`.

```
cd ~/GSORB_SLAM
./Examples/RGB-D/rgbd_tum ~/GSORB_SLAM/Vocabulary/ORBvoc.txt ~/GSORB_SLAM/Examples/RGB-D/tum/TUM1.yaml ~/datasets/rgbd_dataset_freiburg1_desk/ ~/GSORB_SLAM/Examples/RGB-D/associations/fr1_desk.txt
```

💡 We provide a script (located at `~/GSORB_SLAM/scripts/run_*.sh`) to run the process—simply update the dataset path.

```
bash scripts/run_tum.sh
```

Run all datasets `bash run.sh`. The results are saved in `~/GSORB_SLAM/experiments`.

### 📊Evaluation

You need to modify the YAML file: `Dataset.name` specifies the result folder, and `Dataset.path` specifies the ground truth path. Then,

```
python scripts/replay.py --yamlPath "your_yaml_path"
```

If you are using the TUM dataset, you also need to add the `--tumAss` parameter.

### 🎓Citation

If you find our code/work useful in your research, please consider citing the following:

```
@ARTICLE{11091447,
  author={Zheng, Wancai and Yu, Xinyi and Rong, Jintao and Ou, Linlin and Wei, Yan and Zhou, Libo},
  journal={IEEE Robotics and Automation Letters}, 
  title={GSORB-SLAM: Gaussian Splatting SLAM benefits from ORB features and Transmittance information}, 
  year={2025},
  pages={1-8},
  doi={10.1109/LRA.2025.3592066}}
```

