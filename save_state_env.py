import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data
from config import MAX_GOAL_DIST

from utilities import Models, Camera, YCBModels
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm
from math import pi
import gym
from gym import spaces
import os
from config import *
module_path = os.path.dirname(__file__)

class FailToReachTargetError(RuntimeError):
    pass


class Throwing(gym.core.GoalEnv):

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, models: Models, camera=None, vis=False) -> None:
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        p.setTimeStep(self.SIMULATION_STEP_DELAY)
        plane = module_path + "/urdf/plane.urdf"
        self.planeID = p.loadURDF(plane)

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # adjusting using sliders to tune parameters (name of the parameter,range,initial value)
        # self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        # self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        # self.zin = p.addUserDebugParameter("z", 0, 1., 0.5)
        # self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        # self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi/2)
        # self.yawId = p.addUserDebugParameter("yaw", -np.pi/2, np.pi/2, np.pi/2)
        # self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

        # initialize ee position
        self.xin = 0
        self.yin = 0
        self.zin = 0.4
        self.rollId = 0
        self.pitchId = np.pi/2
        self.yawId = np.pi/2
        self.gripper_opening_length_control = 0.04

        self.base_obs = np.array([-1.5100624900539077, -1.65785884485203, 0.7612879544266011, -0.6350743593337471, -1.5846764533575841, -1.509362875745434, 0.00027636869794319665, -0.04541021123570039, 0.0015197028348488434, 0.0023324495708379788, -0.3580057611600621, 0.01028467203242047, 0.019016431759689482, -2.9087556394415244, -0.004867098280624094, 3.200331264838743, -0.0037562649695422543, -0.004102172405176071, -0.012925618014997975, -0.004635814119956488, 0.0041808403666174915, 0.007176108461127012, 0.0, 0.00018500192411713412, -0.152517673826067, 0.19584308230704414, 1.2257761821941566])

        # env.action_space.high[0]
        self.goal_dim = 3
        self.state_dim = len(self.base_obs)
        self.obs_low = np.array([-1] * self.state_dim)
        self.obs_high = np.array([1] * self.state_dim)
        self.observation_space = spaces.Dict(dict(
            desired_goal    =spaces.Box(0, 1, shape= (self.goal_dim,), dtype='float32'),
            achieved_goal   =spaces.Box(0, 1, shape= (self.goal_dim,), dtype='float32'),
            observation     =spaces.Box(0, 1, shape= (self.state_dim + 1,), dtype='float32'),
        ))

        self.action_space = spaces.Box(-1, 1, shape= (9,), dtype='float32')

        self.cube_position = [-1.8, 0, 0]
        self.cube_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.collision_cube_position = [-1.8, 0, 0.3]

        self.ball_base_position = [0, -0.3, 0.3]

        self.ball_position = [-0.0, -0.15, 0.35]
        self.ball_orientation = p.getQuaternionFromEuler([0, 0, 0])
        '''
        self.collision_cube = p.loadURDF("./urdf/block/cube_0.3/cube2.urdf", 
                                    self.collision_cube_position, self.cube_orientation,
                                    useFixedBase=False,
                                    flags = p.URDF_USE_SELF_COLLISION)
        ''' 
        # self.cube = p.loadURDF("./urdf/block/cube_0.3/cube.urdf", 
        #                             self.cube_position, self.cube_orientation,
        #                             useFixedBase=False,
        #                             flags = p.URDF_USE_SELF_COLLISION)
        
        self.ball_base = p.loadURDF("./urdf/objects/table.urdf", 
                                    self.ball_base_position, self.cube_orientation,
                                    useFixedBase=True,
                                    flags = p.URDF_USE_SELF_COLLISION)
        
        self.ball = p.loadURDF("./urdf/ball_test.urdf", self.ball_position, self.ball_orientation)
        # For calculating the reward
        '''
        self.state_space = {
            'state' :,
            'achieved_goal' :,
            'desired_goal' :
        }
        '''
        self.goal = self.sample_goal()
        self.integration_step = 0.01
        p.setTimeStep(self.integration_step)

        self.distance_threshold = 0.4
        self.reward_type = 'sparse'
        self.old_reset()
        self.saved_state = p.saveState()
        p.restoreState(self.saved_state)
        
        # self.goal_vis = p.loadURDF('./urdf/block/brick_0.3/brick.urdf', self.goal, p.getQuaternionFromEuler([0, 0, 0]))

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def uptown_funk(self, time = 120):
        # STOP! WAIT A MINUTE
        steps = int(time/self.integration_step)
        for _ in range(steps):  
            self.step_simulation()

    def noise_force(self):
        pass

    def read_debug_parameter(self):
        # FOR ENV DEBUGGING JUST IGNORE THIS
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return x, y, z, roll, pitch, yaw, gripper_opening_length
    
    def is_success(self, achieved_goal, desired_goal):
        d = self.goal_distance(achieved_goal, desired_goal)
        print(d)
        return (d<self.distance_threshold)
    
    def add_random_force_to_object(self):
        # pass
        # random_force_range_x = 0
        # random_force_range_y = 0
        # random_force_range_z = 1
        # forceX = np.random.rand() * random_force_range_x - random_force_range_x/2
        # forceY = np.random.rand() * random_force_range_y - random_force_range_y/2
        # forceZ = np.random.rand() * random_force_range_z - random_force_range_z/2
        # random_force = [forceX,forceY,forceZ]
        random_force = self.wind_vector.tolist()
        ball_pos = self.get_ball_obs()
        if ball_pos[-1] > 0.06:
            p.applyExternalForce(self.ball, -1, random_force, ball_pos, flags = p.WORLD_FRAME)

    def step(self, action, control_method='end'):
        """
        throw: (x, y, z, roll, pitch, yaw, gripper_opening_length, release_time, car_velocity)
        move: (x, y, z, roll, pitch, yaw, gripper_opening_length, release_time, car_velocity)
        """
        assert control_method in ('end', 'joint')
        

        gas = action[-1]*MAX_SPEED
        grip = (action[7]+1)/2*0.085
        self.robot.move_ugv(gas)
        base_pos, _ = p.getBasePositionAndOrientation(self.robot.id)
        # base_pos x ,y ,z
        # import pdb
        # pdb.set_trace()
        absolute_pos = action[:6] 
        absolute_pos[:3] += np.array(base_pos)
        absolute_pos[3:6] = (absolute_pos[3:6] + 1/2)*pi

        self.robot.move_ee(absolute_pos, control_method)
        delta_t = .2
        release_time = (action[-2] + 1)/2* delta_t
        if release_time > delta_t:
            self.uptown_funk(delta_t)

        elif release_time <= delta_t:
            self.uptown_funk(release_time)
            self.robot.move_gripper(grip)
            self.uptown_funk(delta_t-release_time)
        # self.uptown_funk(release_time)
        
        # ratio = np.random.rand()
        # if ratio<0.3:
        #     self.add_random_force_to_object()
        self.add_random_force_to_object()
        # self.uptown_funk(100)
        state = self.get_state()
        achieved_goal = state['achieved_goal']
        desired_goal = state['desired_goal']
        reward = self.compute_reward(achieved_goal, desired_goal)

        done = True if reward == 0 else False
        # reward = self.compute_reward()
        info = {"is_success": done}
        return state, reward, done, info

    def update_reward(self):

        """
        realtime height check
        to be implemented
        """
        reward = -1
        if self.box_collide == True:
            print("SUCCESS!")
            reward = 0
        return reward

    def get_ball_obs(self):
        position, ori = p.getBasePositionAndOrientation(self.ball)
        # velocity = []
        # velocity, angular_velocity = p.getBaseVelocity(self.ball)
        # return dict(position = position, velocity = velocity)
        # position = position
        return position

    def sample_goal(self):
        sample_x = -0.5 - MAX_GOAL_DIST * np.random.random()
        sample_y = (0.8 - 1.6 * np.random.random()) + 0.5
        # sample_z = 0
        # goal = (sample_x, sample_y, sample_z)
        goal = (sample_x, sample_y)
        return goal

    def get_state(self):
        # ROBOT STATE
        joint_position, joint_velocity, ee_pos = self.robot.get_joint_obs()
        obs = joint_position + joint_velocity + ee_pos
        
        # ACHIEVED GOAL
        achieved_goal = self.get_ball_obs()[:2]

        # DESIRED GOAL
        desired_goal = self.goal
        ag = np.array(achieved_goal[:2])
        state = {
            'observation': np.concatenate([np.array(obs) - self.base_obs, ag]),
            'achieved_goal': ag,
            'desired_goal': np.array(desired_goal[:2])
        }
        return state

    def get_observation(self):
        # USING THE CAMERA TO OBTAIN THE OBSERVATION
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def goal_distance(self, goal_a, goal_b):
        goal_a = np.array(goal_a)
        goal_b = np.array(goal_b)
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
    
    def compute_reward(self, achieved_goal, goal, info=None):
        d = self.goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d
        
    def reset_env(self):
        
        p.resetBasePositionAndOrientation(self.ball, self.ball_position, self.ball_orientation)
        # p.resetBasePositionAndOrientation(self.cube, self.cube_position, self.cube_orientation)
        # after initialize the position of the ball and cube, grasp the ball as initial state
        
        self.robot.move_ee((0, -0.15, 0.75, 1.570796251296997, 1.570796251296997, 1.570796251296997),'end')
        self.uptown_funk(0.5)
        self.robot.open_gripper()
        self.robot.move_ee((0, -0.15, 0.50, 1.570796251296997, 1.570796251296997, 1.570796251296997),'end')        
        self.uptown_funk(0.5)
        self.robot.close_gripper()
        self.uptown_funk(0.5)

        self.robot.move_ee((0, -0.15, 0.7, 1.570796251296997, 1.570796251296997, 1.570796251296997),'end')
        self.uptown_funk(0.5)

    def old_reset(self):
        self.goal = self.sample_goal()
        ### goal visualization
        # p.removeBody(self.goal_vis)
        self.goal_vis = p.loadURDF('./urdf/block/brick_0.3/brick.urdf', self.goal  +(0,), p.getQuaternionFromEuler([0, 0, 0]))
        self.robot.reset()
        self.reset_env()
        state = self.get_state()
        return state

    def reset(self):        
        self.goal = self.sample_goal()
        p.restoreState(self.saved_state)
        if np.random.rand() < WIND_CHANCE:
            self.wind_vector = np.random.normal(np.zeros(3), np.array([WIND_FORCE, WIND_FORCE, 0]))
        else:
            self.wind_vector = np.zeros(3)
        state = self.get_state()
        return state

    def close(self):
        p.disconnect(self.physicsClient)


def make_throwing_env():
    import os
    from robot import UR5Robotiq85, HuskyUR5
    ycb_models = YCBModels(os.path.join('./data/ycb', '**', 'textured-decmp.obj'),)
    camera = Camera((4, 0, 1),
                        (0, -0.7, 0),
                        (0, 0, 1),
                        0.1, 5, (320, 320), 40)
    robot = HuskyUR5((-0.2, 0.5, 0.5), (0, 0, 0))
    return  Throwing(robot, ycb_models, camera, vis=False)
    # return  Throwing(robot, ycb_models, camera, vis=True)

if __name__ == "__main__":
    env = make_throwing_env(vis=True)
    env.reset()
    for i in range(50):
        state, reward, done, info = env.step(np.array((-0, -0, 0.9, 0, 0, 0, 0.1, 0.5, -.5)))
# count = 0

# env.reset()

# while count < 10000:
#     # env.step(env.read_debug_parameter(),'end')
#     # env.step((-0.6, -0.1, 0.9, 1.570796251296997, 1.570796251296997, 1.570796251296997, 0.1),'end')
#     state, reward, done, info = env.step(np.array((-0, -0, 0.9, 0, 0, 0, 0.1, 0.5, -.5)))
#     state, reward, done, info = env.step(np.array((-0, -0, 0.8, 0, 0, 0, 0.1,-0.5, -0.5)))
#     count = count + 1
#     print("+==========================================+")
#     print(state)