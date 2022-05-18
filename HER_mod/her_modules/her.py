import numpy as np
from HER_mod.rl_modules.hyperparams import COLLISION_COST

def bound(x):
    clip_x = np.clip(x, -1, 1)
    diff = x - clip_x
    # import pdb
    # pdb.set_trace()
    if (diff**2 < .01).all():
        return x
    else: 
        return bound(clip_x - diff)
        # return clip_x - diff#bound(clip_x - diff)

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

        self.base_count = 10000
        self.count = self.base_count

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        # future_offset = np.random.geometric(.1, size=batch_size) % (T - t_samples)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['policy_g'] = transitions['g'].copy()
        transitions['g'][her_indexes] = future_ag 
        # transitions['policy_g'] = transitions['g'].copy()


        transitions['t']  = future_offset
        self.count += 1
        # self.base_std = (self.count/10000)**.5
        # if self.count % 1000==0:
        #     print(self.count)

        # if transitions['g'].shape == transitions['ag_next'].shape:
        #     # if np.random.rand() > (10000/self.count)**.5: 
        #     #     rands = (np.random.random_sample(transitions['g'].shape)*2-1)
        #     #     transitions['g'] = rands
        #     # # alpha = .7
        #     # # rands = (np.random.random_sample(transitions['g'].shape)*2-1)
        #     rands = (np.random.standard_normal(transitions['g'].shape))
        #     # rands[...,:2] *= .0
        #     # rands[...,2:] *= .0
        #     rands[...,:2] *= .0
        #     # rands[...,2:] *= .3
        #     rands[...,2:] *= 1
        #     # rands[...,:2] *= .1
        #     # rands[...,2:] *= .7#*self.base_std

        #     # if np.random.rand() > (self.base_count/self.count): 
        #     #     rands = (np.random.random_sample(transitions['g'].shape)*2-1)
        #         # transitions['g'] = rands
        #     # # rands=rands*0
        #     transitions['g'] = bound(transitions['g'] + rands)
        #     # # transitions['g'][...,2:] = (alpha*transitions['g'] + (1-alpha)*rands)[...,2:]
        #     # # transitions['g'][...,2:] = (transitions['g'] + (1-alpha)*rands)[...,2:]
        # to get the params to re-compute reward
        collided = transitions['col']
        rewards = self.reward_func(transitions['ag_next'], transitions['g'], None) - collided*(COLLISION_COST)
        transitions['r'] = np.expand_dims(rewards, 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        #import pdb
        #pdb.set_trace()

        return transitions

    # def sample_ddpg_transitions(self, episode_batch, batch_size_in_transitions):
    #     T = episode_batch['actions'].shape[1]
    #     rollout_batch_size = episode_batch['actions'].shape[0]
    #     batch_size = batch_size_in_transitions
    #     # select which rollouts and which timesteps to be used
    #     episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
    #     t_samples = np.random.randint(T, size=batch_size)
    #     transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}


    #     rands = (np.random.standard_normal(transitions['g'].shape))
    #     rands[...,:2] *= .0
    #     rands[...,2:] *= .3#*self.base_std

    #     # if np.random.rand() > (self.base_count/self.count): 
    #     #     rands = (np.random.random_sample(transitions['g'].shape)*2-1)
    #         # transitions['g'] = rands
    #     # # rands=rands*0
    #     transitions['g'] = bound(transitions['g'] + rands)

    #     # if np.random.rand() < .5: 
    #     rands = (np.random.random_sample(transitions['g'].shape)*2-1)
    #     transitions['g'] = rands

    #     # to get the params to re-compute reward
    #     transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
    #     transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

    #     return transitions

    def sample_ddpg_transitions(self, episode_batch, batch_size_in_transitions):
        return self.sample_her_transitions(episode_batch, batch_size_in_transitions)
    #     T = episode_batch['actions'].shape[1]
    #     rollout_batch_size = episode_batch['actions'].shape[0]
    #     batch_size = batch_size_in_transitions
    #     # select which rollouts and which timesteps to be used
    #     episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
    #     t_samples = np.random.randint(T, size=batch_size)
    #     transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
    #     # her idx
    #     her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p*(self.base_count/self.count)**.5)
    #     future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
    #     future_offset = future_offset.astype(int)
    #     # future_offset = np.random.geometric(.1, size=batch_size) % (T - t_samples)
    #     future_t = (t_samples + 1 + future_offset)[her_indexes]
    #     # replace go with achieved goal
    #     future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]

    #     # random_ag_indexes = np.where(np.random.uniform(size=future_ag.shape[0]) < 1.1 - (self.base_count/self.count)**.4)

    #     # future_ag[random_ag_indexes] = (np.random.random_sample(future_ag[random_ag_indexes].shape)*2-1)

    #     transitions['g'][her_indexes] = future_ag 

    #     transitions['t']  = future_offset
    #     self.count += 1
    #     self.base_std = (self.count/10000)**.5
    #     # if self.count % 1000==0:
    #     #     print(self.count)

    #     if transitions['g'].shape == transitions['ag_next'].shape:
    #         # if np.random.rand() > (10000/self.count)**.5: 
    #         #     rands = (np.random.random_sample(transitions['g'].shape)*2-1)
    #         #     transitions['g'] = rands
    #         # # alpha = .7
    #         # rands = (np.random.random_sample(transitions['g'].shape)*2-1)
    #         rands = (np.random.standard_normal(transitions['g'].shape))
    #         rands[...,:2] *= .1
    #         rands[...,2:] *= .7#*self.base_std

    #         # if np.random.rand() > (self.base_count/self.count): 
    #         #     rands = (np.random.random_sample(transitions['g'].shape)*2-1)
    #             # transitions['g'] = rands
    #         # # rands=rands*0
    #         transitions['g'] = bound(transitions['g'] + rands)
    #         # transitions['g'][...,2:] = rands[...,2:]
    #         # transitions['g'] = bound(transitions['g'] + rands)
    #         # # transitions['g'][...,2:] = (alpha*transitions['g'] + (1-alpha)*rands)[...,2:]
    #         # # transitions['g'][...,2:] = (transitions['g'] + (1-alpha)*rands)[...,2:]


    #     transitions['g'][her_indexes] = future_ag 
        
    #     # to get the params to re-compute reward
    #     transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
    #     transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

    #     return transitions
