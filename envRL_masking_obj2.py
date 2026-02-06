
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete, Sequence
from gymnasium.wrappers import FlattenObservation

# import helpers
import numpy as np
import random
import os
import tensorboard
import pygame
import pylab
from pygame.locals import *
import time
import copy

# import stable_baselines3
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# import sb3_contrib
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
# render related


# routing related
import networkx as nx
from statistics import mode
import matplotlib
import matplotlib.pyplot as plt
import main
import NSGA
from main import Graph, FishGround, FishFactory, WayPoint, Start, End, Position, great_circle_distance, \
    scgraph_distance, pathRisk, pathCatch, pathGain, pathFuel, pathDistance, Ship, dynamic_routing


# Create a class function for making training environment
# Objective: Get the most catch


G = nx.DiGraph()

graph1 = Graph(G)
loc1 = FishGround(name='FG1', location=Position(30, 15), fishstock={'Cod': {'quantity': 100, 'timewindow': (1, 2)}})
loc2 = FishGround(name='FG2', location=Position(30, 50), fishstock={'Cod': {'quantity': 500, 'timewindow': (1, 2)}})
loc3 = FishGround(name='FG3', location=Position(60, 50), fishstock={'Cod': {'quantity': 300, 'timewindow': (1, 2)}})
start = Start(name='S', location=Position(10, 10))
end = End(name='E', location=Position(80, 80))
ff1 = FishFactory(name='FF1', location=Position(50, 60))
ff2 = FishFactory(name='FF2', location=Position(70, 50))

locations = [loc1, loc2, loc3, start, end, ff1, ff2]
graph1.addlocations_2(locations=locations)
ship = Ship(targeted_fish='Cod', targeted_quantity=1000, weight=1000)

matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg


class Routing(Env):
    def __init__(self, Graph: Graph, Ship: Ship, start):
        self.ship = copy.deepcopy(Ship)
        self.graph = Graph
        self.Footprint = nx.DiGraph()
        self.Footprint.add_node(start)
        nodeattr = self.graph.graph.nodes[start].copy()
        nx.set_node_attributes(self.Footprint, {start: nodeattr})
        self.successorsLst = nx.dfs_successors(G=self.graph.graph, source=start, depth_limit=1)[start]
        self.action_space = Discrete(7)
        # old_action_space=Dict({'location_to_go': Discrete(len(self.successorsLst)), 'speed': Box(low=0, high=30, shape=(1,)), 'duration': Box(0, 10, shape=(1,))})
        self.observation_space = Dict({'route': Discrete(7),
                                       'pathCatch': Box(low=0, high=30000, shape=(1,), dtype=float)})
        self.current_place = start
        locations.index(start)
        self.state = {'route': locations.index(start), 'pathCatch': 0}
        self.route_state = {'footprint': self.Footprint, 'route': [start]}
        self.previous_place = None
        self.route_length = 10

    def step(self, action):
        # print(action[0])
        #print('successors:', [loc.name for loc in self.successorsLst])
        location = locations[action]
        # print(location.name)
        self.previous_place = self.current_place
        # Update state, including Footprint, route, speed and duration
        self.current_place = location
        self.route_state['route'].append(self.current_place)
        nodeattr = self.graph.graph.nodes[self.current_place].copy()
        self.Footprint.add_node(self.current_place)
        nx.set_node_attributes(self.Footprint, {self.current_place: nodeattr})
        self.Footprint.add_edge(self.previous_place, self.current_place)
        edgeattr = self.graph.graph.edges[(self.previous_place, self.current_place)].copy()
        nx.set_edge_attributes(self.Footprint, {(self.previous_place, self.current_place): edgeattr})
        #print('current_place', self.current_place.name)
        # rewards
        if isinstance(self.current_place, FishGround):
            fishtype = self.ship.targeted_fish
            newcatch = self.ship.update_catch(self.current_place)
            stock = self.current_place.fishstock
            stock[fishtype]['quantity'] -= newcatch
            self.current_place.update_stock(stock)
            self.graph.update_node_attribute(self.current_place, 'fishstock', stock)
        else:
            newcatch = 0
        self.state = {'route': action, 'pathCatch': newcatch}
        reward = newcatch

        info = {}

        self.route_length -= 1
        # conditions to end episode
        if isinstance(self.current_place, End):
            #reward = reward
            done = True
        elif self.route_length <= 0:
            #reward = -1000
            done = True
        else:
            done = False
        # print('steps', self.route_length)
        # print('obs', self.state)

        return self.state, reward, done, False, info

    def render(self):
        fig = pylab.figure(figsize=[4, 4],  # Inches
                           dpi=200,  # 100 dots per inch, so the resulting buffer is 400x400 pixels
                           )
        fig.gca()
        self.graph.plot_graph()
        pos = {n: [n.position[1], n.position[0]] for n in list(self.Footprint.nodes())}
        nx.draw_networkx(self.Footprint, with_labels=False, pos=pos, node_color='red', edge_color='red',
                         font_color='red')
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_argb()
        window = pygame.display.set_mode((800, 800), DOUBLEBUF)
        screen = pygame.display.get_surface()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "ARGB")
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    def get_obs(self):
        pass

    def get_info(self):
        self.info = {}
        return self.info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        loc1.fishstock['Cod']['quantity'] = 100
        loc2.fishstock['Cod']['quantity'] = 500
        loc3.fishstock['Cod']['quantity'] = 300
        self.ship = copy.deepcopy(ship)
        self.Footprint = nx.DiGraph()
        self.Footprint.add_node(start)
        nodeattr = self.graph.graph.nodes[start].copy()
        nx.set_node_attributes(self.Footprint, {start: nodeattr})
        self.successorsLst = nx.dfs_successors(G=self.graph.graph, source=start, depth_limit=1)[start]
        self.action_space = Discrete(7)
        self.observation_space = Dict({'route': Discrete(7), 'pathCatch': Box(low=0, high=30000, shape=(1,), dtype=float)})
        self.current_place = start
        self.route_state = {'footprint': self.Footprint, 'route': [start]}
        self.previous_place = None
        self.route_length = 10
        self.state = {'route': locations.index(start), 'pathCatch': 0}
        info = self.get_info()
        return self.state, info

    def retrieve_location(self, action):
        location = self.successorsLst[action[0]]
        return

    def action_masks(self):
        if not isinstance(self.current_place, End):
            successors = nx.dfs_successors(G=self.graph.graph, source=self.current_place, depth_limit=1)
            #print('current_place', self.current_place.name)
            self.successorsLst = successors[self.current_place]
            # code below exclude fish ground where its stock is 0.
            locLst = []
            for loc in self.successorsLst:
                if isinstance(loc, FishGround):
                    if loc.fishstock['Cod']['quantity'] <= 0:
                        locLst.append(loc)
            newLst = list(filter(lambda loc: loc not in locLst, self.successorsLst))
            self.successorsLst = newLst
            location_mask = [1 if loc in newLst else 0 for loc in locations]
        else:
            successors = nx.dfs_successors(G=self.graph.graph, source=start, depth_limit=1)
            self.successorsLst = successors[start]
            location_mask = [1 if loc in self.successorsLst else 0 for loc in locations]
        #print('location mask:', location_mask)
        location_mask = np.array(location_mask, dtype=np.int8)
        action_mask = location_mask
        #print(action_mask)
        return action_mask

def mask_fn(env):
    return env.action_masks()
envR = Routing(Graph=graph1, Ship=ship, start=start)
envR = ActionMasker(envR, mask_fn)

#Specify location for log and model file
Objective = 'Obj2'

#Test training environment
# episodes = 5
# score_list = []
# for episode in range(1, episodes +1):
#     print('episode:{}'.format(episode))
#     done = False
#     obs, _ = envR.reset()
#     envR.render()
#     pygame.time.delay(1000)
#     score = 0
#     length = 10
#     while not done:
#         action = envR.action_space.sample(mask=envR.action_masks())
#         print('action chosen:', action)
#         obs, reward, done, truncated, info = envR.step(action)
#         envR.render()
#         pygame.time.delay(1000)
#         score += reward
#     score_list.append(reward)
#     print('episode:{} reward:{}'.format(episode, reward))
# print(np.mean(score_list))
# print(np.std(score_list))
# envR.close()

# #Train an agent
# timesteps = [1000000] # 20000, 100000, 1000000, 3000000, 5000000, 10000000
# for timestep in timesteps:
#     total_timesteps=timestep
#     start_time = time.time()
#     log_path = os.path.join('Training', 'Logs', Objective, f'{Objective}_{total_timesteps}stepModel')
#     model_path = os.path.join('Training', 'Model', Objective, f'{Objective}_{total_timesteps}stepModel')
#     model = MaskablePPO("MultiInputPolicy", envR, verbose=1, tensorboard_log=log_path, seed=100)
#     model.learn(total_timesteps=total_timesteps)
#     finish_time = time.time()
#     time_used = finish_time-start_time
#     print('Time spent for training:', time_used)
#     #save the trained model
#     model.save(model_path)
#
# # Load a trained agent
# total_timesteps=5000000
# model_path = os.path.join('Training', 'Model', Objective, f'{Objective}_{total_timesteps}stepModel')
# model = MaskablePPO.load(model_path, env=envR)
#
#
# # Test the trained agent
# vec_env = model.get_env()
# obs  = vec_env.reset()
# print('Initial observation',obs)
# action, _ = model.predict(obs, action_masks=envR.action_masks())
# print('chosen action by agent according to observation', action)
# obs, reward, done, truncated = vec_env.step(action)
# print('step output',obs, reward, done, truncated)
#
# # Run some episodes to see how the trained agent performs
# episodes = 10
# score_list = []
# for episode in range(1, episodes + 1):
#     print('episode:{}'.format(episode))
#     done = False
#     obs = vec_env.reset()
#     envR.render()
#     pygame.time.delay(100)
#     print('Observation output from initial',obs)
#     score = 0
#     length = 10
#     while not done:
#         print('input_obs', obs)
#         action, _ = model.predict(obs, action_masks=envR.action_masks()) #, action_masks=envR.action_masks()
#         print('action chosen:', action)
#         obs, reward, done, truncated = vec_env.step(action)
#         envR.render()
#         pygame.time.delay(100)
#         print('output_obs', obs, reward, done, truncated) #observation is the same as state
#         score += reward
#     score_list.append(reward)
# print(np.mean(score_list))
# print(np.std(score_list))
# envR.close()

