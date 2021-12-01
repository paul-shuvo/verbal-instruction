## Planar Pose Estimation

A ROS package for a simple object detection and planar pose estimation method for textured objects.

[![CodeFactor](https://www.codefactor.io/repository/github/paul-shuvo/verbal-instruction/badge)](https://www.codefactor.io/repository/github/paul-shuvo/verbal-instruction)
[![Build Status](https://app.travis-ci.com/paul-shuvo/verbal-instruction.svg?branch=main)](https://app.travis-ci.com/paul-shuvo/verbal-instruction)
![](https://img.shields.io/badge/ROS-Noetic%20%7C%20Melodic%20%7C%20Kinetic-blue)
![](https://img.shields.io/badge/Python-2.7%20%7C%203.3+-green)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)




<!-- ## Table of contents -->
- [Planar Pose Estimation](#planar-pose-estimation)
  - [Dependencies](#dependencies)
  - [Install](#install)
  - [Run](#run)
  - [Published Topics](#published-topics)
  - [Demo](#demo)
  - [TODO](#todo)

### Dependencies

- ROS Noetic/Melodic/Kinetic
- Python 2.7, 3.3+ for ROS Noetic (should support Python 3 for other ROS Distros as well)
- OpenCV (Tested on opencv 4, should work with other opencv versions as well)

### Install
For new projects:

Run the following

```
sudo apt-get install libasound-dev
sudo apt-get install portaudio19-dev
pip install pyaudio --user
```

Then,

```
mkdir catkin_ws/src
cd catkin_ws/src
git clone https://github.com/paul-shuvo/verbal-instruction.git
cd verbal-instruction
pip install -r requirements.txt
cd ../../..
catkin_make
```

For existing project:

```
cd 'your_project_path'/src
git clone https://github.com/paul-shuvo/verbal-instruction.git
cd verbal-instruction
pip install -r requirements.txt
cd ../../..
catkin_make
```

### Run

This package contains a object detection module and a planar pose estimation module. Planar pose estimation module depends on the object detection module.

To run the object detection module:
```
cd 'your_project_path`
source devel/setup.bash
rosrun verbal-instruction extract_param.py
```

### Published Topics



### Demo



### TODO

- [ ] Do more tests
- [ ] Documentation