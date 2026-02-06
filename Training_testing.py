import numpy as np

import os
import pygame

import time


# import sb3_contrib
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO

import matplotlib.backends.backend_agg as agg

# Import some environments which have different objectives and/or rewards function
import envRL_masking_obj1,envRL_masking_obj2, envRL_masking_obj3
import envRL_masking_obj4, envRL_masking_obj5, envRL_masking_obj8


#Specify environment, feel free to change to another training environment (from 1, 2, 3, 4, 5, 8)
envR=envRL_masking_obj1.envR
#Specify location for log and model file
Objective = 'Obj1'

#Test training environment
# episodes = 5
# for episode in range(1, episodes +1):
#     print('episode:{}'.format(episode))
#     done = False
#     obs, _ = envR.reset()
#     envR.render()
#     pygame.time.delay(1000)
#     length = 10
#     while not done:
#         action = envR.action_space.sample(mask=envR.action_masks())
#         print('action chosen:', action)
#         obs, reward, done, truncated, info = envR.step(action)
#         envR.render()
#         pygame.time.delay(1000)
#     print('episode:{} reward:{}'.format(episode, reward))
# envR.close()

# #Train an agent
# timesteps = [10000, 1000000, 3000000] # 20000, 100000, 1000000, 3000000, 5000000, 10000000
# for timestep in timesteps:
#     total_timesteps=timestep
#     start_time = time.time()
#     log_path = os.path.join('Training', 'Logs', f'{Objective}_{total_timesteps}stepModel')
#     model_path = os.path.join('Training', 'Model', f'{Objective}_{total_timesteps}stepModel')
#     model = MaskablePPO("MultiInputPolicy", envR, verbose=1, tensorboard_log=log_path, seed=100)
#     model.learn(total_timesteps=total_timesteps)
#     finish_time = time.time()
#     time_used = finish_time-start_time
#     print('Time spent for training:', time_used)
#     #save the trained model
#     model.save(model_path)

# Load a trained agent
total_timesteps=3000000
model_path = os.path.join('Training', 'Model', f'{Objective}_{total_timesteps}stepModel')
model = MaskablePPO.load(model_path, env=envR)
#
#
# Test the trained agent
vec_env = model.get_env()
obs  = vec_env.reset()
print('Initial observation',obs)
action, _ = model.predict(obs, action_masks=envR.action_masks())
print('chosen action by agent according to observation', action)
obs, reward, done, truncated = vec_env.step(action)
print('step output',obs, reward, done, truncated)

# Run some episodes to see how the trained agent performs
episodes = 10
for episode in range(1, episodes + 1):
    print('episode:{}'.format(episode))
    done = False
    obs = vec_env.reset()
    envR.render()
    pygame.time.delay(1000)
    print('Observation output from initial',obs)
    length = 10
    while not done:
        print('input_obs', obs)
        action, _ = model.predict(obs, action_masks=envR.action_masks()) #, action_masks=envR.action_masks()
        print('action chosen:', action)
        obs, reward, done, truncated = vec_env.step(action)
        envR.render()
        pygame.time.delay(1000)
        print('output_obs', obs, reward, done, truncated) #observation is the same as state
envR.close()









