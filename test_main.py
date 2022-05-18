import pybullet as p
import pybullet_data
import gym

import numpy as np
import math
from test_env import ThrowBall
from robot import UR5Robotiq85
from utilities import YCBModels, Camera
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

ycb_models = YCBModels(os.path.join('./data/ycb/', '**', 'textured-decmp.obj'))

camera = Camera((4, 0, 1),
                    (0, -0.7, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)

# camera = None
# robot = Panda((0, 0.5, 0), (0, 0, math.pi))
robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
env = ThrowBall(robot, ycb_models, camera, vis=True)
env.reset()

current_state = 0
state_t = 0
control_dt = 1./240

while True:
    state_t += control_dt
    obs, reward, done, info = env.step(env.read_debug_parameter(), 'end')
