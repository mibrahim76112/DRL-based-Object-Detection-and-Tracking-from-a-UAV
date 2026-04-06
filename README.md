# DRL-based Object Detection and Tracking from a UAV

A deep reinforcement learning based project for **object detection and tracking from a UAV**.
This repository combines **computer vision**, **drone control**, and **reinforcement learning** to detect a target through vision input and guide a UAV toward the object using learned control actions.


---

## Project Overview

This repository presents a UAV tracking system where an object is detected from image-based bounding box information and the drone is controlled using deep reinforcement learning.

The system uses object detections from a ROS topic and computes the position of the target in the image frame. Based on the error between the image center and the detected object center, the learning agent selects actions to move the UAV and improve tracking performance.

The repository also includes code for different reinforcement learning variants, such as standard DDPG, TD3, and prioritized replay based DDPG.

---

## Features

- UAV target detection and tracking framework
- Reinforcement learning based control
- DDPG implementation
- TD3 implementation
- Prioritized Experience Replay support
- Custom environment and reward design
- ROS based object detection input
- MAVLink based drone movement control
- Research paper included
- Screenshots of results included

---

## Repository Structure

```text
DRL-based-Object-Detection-and-Tracking-from-a-UAV/
│
├── main_ddpg.py                              # Main script for DDPG based tracking
├── main_td3.py                               # Main script for TD3 based tracking
├── main_per.py                               # Main script for prioritized replay training
├── ddpg_torch.py                             # DDPG agent implementation
├── td3_torch.py                              # TD3 agent implementation
├── PER_DDPG.py                               # DDPG with prioritized replay
├── buffer.py                                 # Standard replay buffer
├── PER_buffer.py                             # Prioritized replay buffer
├── networks.py                               # Actor and critic neural networks
├── env.py                                    # Tracking environment and reward logic
├── utils.py                                  # Utility functions
├── Paper.pdf                                 # Project paper or report
├── Screenshot from 2024-05-19 16-11-47.png   # Result screenshot
├── Screenshot from 2024-05-19 16-11-50.png   # Result screenshot
├── Screenshot from 2024-05-19 17-19-59.png   # Result screenshot
└── README.md                                 # Project documentation
```

---

## System Idea

The overall idea of the project is to track a target from a UAV by combining detection and control:

1. An object detector provides bounding box information.
2. The target center is extracted from the bounding box.
3. The difference between the frame center and target center is used as the tracking error.
4. A reinforcement learning agent observes the current tracking state.
5. The agent predicts an action for UAV movement.
6. The UAV is moved using MAVLink commands.
7. A reward function evaluates whether the drone is improving alignment with the target.

---

## Detection and Tracking Pipeline

The project uses a ROS subscriber to receive bounding boxes from the topic:

`/darknet_ros/bounding_boxes`

The detection callback checks detected objects and looks specifically for a **person** class. Once a person is detected, the center of the bounding box is computed and used for tracking control.

This creates a vision-guided loop in which object detection provides the observation signal for the reinforcement learning controller.

---

## Reinforcement Learning Methods Included

This repository contains multiple reinforcement learning approaches for UAV target tracking:

### DDPG
Deep Deterministic Policy Gradient is used for continuous control of the UAV movement.

### TD3
Twin Delayed Deep Deterministic Policy Gradient is included as an improved actor-critic method for more stable learning.

### Prioritized Experience Replay
A prioritized replay based DDPG implementation is also included to improve sample efficiency by replaying more informative transitions.

---

## Environment Design

The custom environment models the tracking task using image-based state information.

The state is based on values such as:

- horizontal error
- vertical error
- distance from image center
- state change across time steps

The reward logic is designed to encourage the UAV to move closer to the detected object and align the target near the image center.
A large positive reward is given when the object becomes well-centered, while a strong negative reward is applied when the target is lost.

---

## Drone Control

The control side of the project uses **pymavlink** to communicate with the drone.
Movement commands are sent through MAVLink local NED messages.

The code also waits for the UAV to enter **GUIDED** mode before running the tracking loop, which shows that the project is designed around practical UAV command flow rather than only offline simulation.

---

## Main Files

### `main_ddpg.py`
Runs the DDPG based tracking loop using ROS detections, the environment, and UAV movement commands.

### `main_td3.py`
Runs the TD3 version of the tracking setup.

### `main_per.py`
Runs the prioritized replay based training or tracking setup.

### `env.py`
Defines the environment state update, reward function, reset logic, and termination conditions.

### `networks.py`
Defines the actor and critic neural network models.

### `buffer.py` and `PER_buffer.py`
Handle standard and prioritized replay memory.

### `Paper.pdf`
Contains the project paper or report.

---

## Tools and Technologies

- Python
- PyTorch
- ROS
- darknet_ros message interface
- pymavlink
- Deep Reinforcement Learning
- UAV control concepts
- Computer vision based target localization

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/mibrahim76112/DRL-based-Object-Detection-and-Tracking-from-a-UAV.git
```

### 2. Install required dependencies

Install the Python packages and ROS related dependencies used in the project.

Typical dependencies include:

- torch
- numpy
- rospy
- pymavlink

You will also need a ROS environment with access to bounding box messages from an object detector.

### 3. Prepare the detection pipeline

Make sure object detections are being published on:

`/darknet_ros/bounding_boxes`

### 4. Connect the UAV or simulator

The code uses MAVLink communication and expects a connection such as:

`udpin:localhost:14550`

### 5. Run one of the training or control scripts

Possible entry scripts include:

- `main_ddpg.py`
- `main_td3.py`
- `main_per.py`

---



