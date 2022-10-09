# social_nav_sim
A ros package for GAZEBO simulation of socially-aware robot navigation methods such as DSRNN(https://github.com/Shuijing725/CrowdNav_DSRNN.git), ORCA, Soical Force, etc.

A number of walking or standing people and various types of static objects are simulated in GAZEBO with random or manually set initial positions and goals.

# Prerequisite
This package has been verified in ubuntu 20.04, ROS Noetic, GAZEBO 11, Python3.8, PyTorch 1.7.1 with CUDA 11.0 and cuDNN 8.0.5.
## NVIDIA Driver, CUDA Toolkit, and cuDNN
Refer to the links below.
NVIDIA Driver: (apt installation recommended) https://phoenixnap.com/kb/install-nvidia-drivers-ubuntu
CUDA Toolkit: 
 - Installation: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation
 - Environment setup after installation: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup
cuDNN: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#cudnn-package-manager-installation-overview
## ROS Noetic(supported in Ubuntu 20.04)
Install ROS Noetic(desktop-full) following the instructions in http://wiki.ros.org/noetic/Installation/Ubuntu.
## Python Packages
Enter the following command in a terminal window.
```
pip3 install numpy lxml
```
## Pytorch
Install PyTorch 1.7.1 using the following command referenced from 
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```
## Python-RVO2
Refer to the link below.
https://github.com/sybrenstuvel/Python-RVO2
## ROS packages
```
suo apt install ros-noetic-velodyne-driver ros-noetic-velodyne-description ros-noetic-velodyne-gazebo-plugins ros-noetic-velodyne-msgs
```
# 
