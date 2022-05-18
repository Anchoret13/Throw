import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch
import itertools
# from rl_modules.multi_goal_env2 import *
from rl_modules.velocity_env import *
# from ..pomp.planners.plantogym import *
from rl_modules.value_map import *
from rl_modules.hooks import *
from rl_modules.tsp import *
from rl_modules.get_path_costs import get_path_costs, get_random_search_costs, method_comparison


"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    # print(params)
    return params

def launch(args, time=True, hooks=[], vel_goal=False, seed=True):
    # create the ddpg_agent
    # env = gym.make(args.env_name)
    # env = MultiGoalEnvironment("MultiGoalEnvironment", time=time)
    env = MultiGoalEnvironment("MultiGoalEnvironment", time=time, vel_goal=vel_goal)#, epsilon=.1/4) 
    # set random seeds for reproduce
    if seed: 
        env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
        random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
        np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
        torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
        if args.cuda:
            torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    # return
    # create the ddpg agent to interact with the environment 
    ddpg_trainer = ddpg_agent(args, env, env_params, vel_goal=vel_goal)
    # if vel_goal: 
    #     ddpg_trainer = ddpg_agent(args, env, env_params, vel_goal=vel_goal)
    # else: 
    #     ddpg_trainer = her_ddpg_agent(args, env, env_params)
    ddpg_trainer.learn(hooks)
    # [hook.finish() for hook in hooks]
    return ddpg_trainer, [hook.finish() for hook in hooks]




if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()

    # agent = launch(args, time=False, hooks=[])#hooks=[DistancePlottingHook()])
    # agent = launch(args, time=True, hooks=[DistancePlottingHook(), PlotPathCostsHook(args)], vel_goal=True)
    # agent = launch(args, time=True, hooks=[DistancePlottingHook(), PlotPathCostsHook(args)], vel_goal=False)
    # try:
    hook_list = [
                # ValueMapHook(target_vels=[[0,0], [.5/2**.5,.5/2**.5]]),
                # DiffMapHook(), 
                # # EmpiricalVelocityValueMapHook(),
                # VelocityValueMapHook(), 
                # GradientDescentShortestPathHook(),
                # GradientDescentShortestPathHook(gd_steps=5),
                # GradientDescentShortestPathHook(gd_steps=10),
                # GradientDescentShortestPathHook(gd_steps=15),
                GradientDescentShortestPathHook(args=([0, 5,10,20,40], False)),
                # GradientDescentShortestPathHook(args=([0,5,10,20,40], True)),
                PlotPathCostsHook()
                ]
    # hook_list = []
    pos_hook_list = [#DiffMapHook(), 
                ValueMapHook(target_vels=[[0,0]]),#target_vels=[[0,0], [.5/2**.5,.5/2**.5]]),
                # GradientDescentShortestPathHook(),
                # GradientDescentShortestPathHook(gd_steps=5),
                # GradientDescentShortestPathHook(gd_steps=10),
                # GradientDescentShortestPathHook(gd_steps=15),
                GradientDescentShortestPathHook(args=([-1], False)),
                # GradientDescentShortestPathHook(args=([-1], True)),
                # PlotPathCostsHook()
                ]
    vel_hook_list = [
                GradientDescentShortestPathHook(args=([0,5,10,20], False)),
                GradientDescentShortestPathHook(args=([0,5,10,20], True)),
                PlotPathCostsHook()
                ]

    # hook_list = []
    # pos_hook_list = []
    # vel_hook_list = []

    train_pos_agent = lambda : launch(args, time=True, hooks=[], vel_goal=False, seed=False)[0]
    train_vel_agent = lambda : launch(args, time=True, hooks=[], vel_goal=True, seed=False)[0]
    # get_path_costs(train_pos_agent, train_vel_agent)
    # train_pos_agent()
    # train_vel_agent()
    # for i in range(10):
    #     args.seed += 1
        # agent, run_times = launch(args, time=True, hooks=hook_list, vel_goal=True, seed=False)
    # agent, run_times = launch(args, time=True, hooks=pos_hook_list, vel_goal=False, seed=False)
    agent, run_times = launch(args, time=True, hooks=hook_list, vel_goal=True, seed=False)
    # plot_cost_by_vel_goal(agent, last_static=True)
    # plot_cost_by_vel_goal(agent, last_static=False)
    # plot_gradient_vector_field(agent)
    # method_comparison(train_pos_agent, train_vel_agent)
    # get_random_search_costs(train_vel_agent, perm_search=False)
    # get_path_costs(train_pos_agent, train_vel_agent, perm_search=False)


    # agent, run_times = launch(args, time=True, hooks=hook_list, vel_goal=False, seed=False)
    # except: 
    #     pass
    # try: 
    #     agent, run_times = launch(args, time=True, hooks=[PlotPathCostsHook(args)], vel_goal=True)
    # except: 
    #     pass
    # agent, run_times = launch(args, time=True, hooks=[ValueMapHook()], vel_goal=True, seed=False)
    # return
    # run_time_list = []
    # for i in range(10):
    #     agent, run_times = launch(args, time=True, hooks=[GradientDescentShortestPathHook(args)], vel_goal=True, seed=False)
    #     run_time_list.append(run_times)

    # # import pdb
    # # pdb.set_trace()

    # run_time_list = np.array(run_time_list).squeeze()
    # mean_time = run_time_list.mean(axis=0)
    # std_time = run_time_list.std(axis=0)
    # ci = 2*std_time/(10**.5)
    # import matplotlib.pyplot as plt
    # steps = np.arange(mean_time.shape[-1])*5
    # plt.plot(steps, mean_time)
    # plt.fill_between(steps, mean_time+ci, mean_time-ci, alpha=.4)
    # plt.show()

    # run_time_list = (run_time_list.transpose() - run_time_list[...,0]).transpose()
    # mean_time = run_time_list.mean(axis=0)
    # std_time = run_time_list.std(axis=0)
    # ci = 2*std_time/(10**.5)
    # import matplotlib.pyplot as plt
    # steps = np.arange(mean_time.shape[-1])*5
    # plt.plot(steps, mean_time)
    # plt.fill_between(steps, mean_time+ci, mean_time-ci, alpha=.4)
    # plt.show()

    # print("vel_goal: True")
    # agent = launch(args, time=True, hooks=[PlotPathCostsHook(args, vel_goal=True)], vel_goal=True)
    # print("vel_goal: False")
    # agent = launch(args, time=True, hooks=[PlotPathCostsHook(args, vel_goal=False)], vel_goal=False)
    # plot_path_costs(args, agent)
    # import pdb
    # pdb.set_trace()
    # agent = launch(args, time=True, hooks=[DistancePlottingHook(), PlotPathCostsHook(args), ValueMapHook()])
 
    # import pdb
    # pdb.set_trace()




    # print(pos)
    # print(goals)


    # # list_dist(pos, goals, l2)

    # print('Analytic solution')
    # min_time, min_path = find_shortest_path(pos, goals, l2)
    # print("Time, Path: ")
    # print(min_time, min_path)
    # print("Time from actual run:")
    # print(path_runner(pos, min_path))

    # print('Experimental solution')
    # min_time, min_path = find_shortest_path(pos, goals, single_goal_run)
    # print("Time, Path: ")
    # print(min_time, min_path)
    # print("Time from actual run:")
    # print(path_runner(pos, min_path))


    # print('Learned solution')
    # min_time, min_path = find_shortest_path(pos, goals, learned_metric)
    # print("Time, Path: ")
    # print(min_time, min_path)
    # # print("Actual time of learned solution: " + str(evaluate_path(pos, min_path, l2)))
    # print("Time from actual run:")
    # print(path_runner(pos, min_path))
