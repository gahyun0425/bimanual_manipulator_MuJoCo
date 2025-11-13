# Bimanual Manipulator MuJoCo

MuJoCo-based **bimanual (dual-arm) manipulator** simulation package, organized as a ROS 2 Python package.

---

## Overview

This repository provides a simulation setup for a bimanual manipulator using the [MuJoCo](https://mujoco.org/) physics engine.

The package includes:

- MuJoCo model files for the bimanual robot
- ROS 2 launch files to start the simulation and related nodes
- Python package with example scripts / nodes

Use this package as a base for research or development involving dual-arm manipulation in simulation.

---

## Bimanual Manipulator Tools

Bimanual robot model tools for MuJoCo-based motion planning & simulation.

### Features

- Collision checker (MoveIt planning scene)
- IK solver (TRAC IK)
- Motion planning (RRT Connect)

---

## Build and Source Workspace
---

### Workspace root directory
```bash
cd ~/bimanual_ws
```
### Source environment
```bash
source install/setup.bash
colcon build
```

## Run Node
```bash
ros2 launch new_bimanual_pkg bimanual_launch.py
```