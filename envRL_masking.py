
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

G = nx.DiGraph()

graph1 = Graph(G)
loc1 = FishGround(name='FG1', location=Position(30, 15), fishstock={'Sei': {'quantity': 100, 'timewindow': (1, 2)}})
loc2 = FishGround(name='FG2', location=Position(30, 50), fishstock={'Sei': {'quantity': 500, 'timewindow': (1, 2)}})
loc3 = FishGround(name='FG3', location=Position(60, 50), fishstock={'Sei': {'quantity': 300, 'timewindow': (1, 2)}})
start = Start(name='S', location=Position(10, 10))
end = End(name='E', location=Position(80, 80))
ff1 = FishFactory(name='FF1', location=Position(50, 60))
ff2 = FishFactory(name='FF2', location=Position(70, 50))

locations = [loc1, loc2, loc3, start, end, ff1, ff2]
graph1.addlocations_2(locations=locations)
ship = Ship(targeted_fish='Sei', targeted_quantity=800, weight=1000)
speed_space = range(1, 200, 1)
duration_space = range(1, 72)

objectives = {'minimize': ['fuel'], 'maximize': ['catch', 'risk']}
constraints = {'total catch': {'low': 600, 'include_low': True, 'high': 1000, 'include_high': True}}
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg


class Routing(Env):
    def __init__(self, Graph: Graph, Ship: Ship, start):
        self.ship = Ship
        self.graph = Graph
        self.Footprint = nx.DiGraph()
        self.Footprint.add_node(start)
        nodeattr = self.graph.graph.nodes[start].copy()
        nx.set_node_attributes(self.Footprint, {start: nodeattr})
        self.successorsLst = nx.dfs_successors(G=self.graph.graph, source=start, depth_limit=1)[start]
        self.action_space = MultiDiscrete([100, 100, 100])
        # old_action_space=Dict({'location_to_go': Discrete(len(self.successorsLst)), 'speed': Box(low=0, high=30, shape=(1,)), 'duration': Box(0, 10, shape=(1,))})
        self.observation_space = Dict({'route': Discrete(100), 'Speed': Discrete(100), 'Duration': Discrete(100),
                                       'pathDistance': Box(low=0.0, high=30000000.0, shape=(1,), dtype=float)})
        self.current_place = start
        locations.index(start)
        self.state = {'route': locations.index(start), 'Speed': 0, 'Duration': 0, 'pathDistance': 0.0}
        self.route_state = {'footprint': self.Footprint, 'route': [start], 'Speed': [], 'Duration': []}
        self.previous_place = None
        self.route_length = 10

    def step(self, action):
        print('Action', action[0])
        print('successors:', [loc.name for loc in self.successorsLst])
        location = locations[action[0]]
        # print(location.name)
        self.previous_place = self.current_place
        if location in self.successorsLst:
            # Update state, including Footprint, route, speed and duration
            self.current_place = location
            self.route_state['route'].append(self.current_place)
            self.route_state['Speed'].append(action[1])
            self.route_state['Duration'].append(action[2])
            self.Footprint.add_edge(self.previous_place, self.current_place)
            edgeattr = self.graph.graph.edges[(self.previous_place, self.current_place)].copy()
            edgeattr['Speed'] = action[1]
            nx.set_edge_attributes(self.Footprint, {(self.previous_place, self.current_place): edgeattr})
            nodeattr = self.graph.graph.nodes[self.current_place].copy()
            nodeattr['Duration'] = action[2]
            self.Footprint.add_node(self.current_place)
            nx.set_node_attributes(self.Footprint, {self.current_place: nodeattr})
            if not isinstance(self.current_place, End):
                successors = nx.dfs_successors(G=self.graph.graph, source=self.current_place, depth_limit=1)
                self.successorsLst = successors[self.current_place]
            #print('current_place', self.current_place.name)
            # rewards
            path = self.route_state['route']
            distance = pathDistance(G=self.Footprint, path=path)
            self.state = {'route': action[0], 'Speed': action[1], 'Duration': action[2], 'pathDistance': distance}
            reward = -distance
        else:
            self.state = {'route': locations.index(self.previous_place), 'Speed': action[1], 'Duration': action[2],
                          'pathDistance': 0.0}
            reward = -100000000
        info = {}

        self.route_length -= 1
        # conditions to end episode
        if self.route_length <= 0 or isinstance(self.current_place, End):
            done = True
        else:
            done = False
        # print('steps', self.route_length)
        # print('obs', self.state)
        truncated=False
        return self.state, reward, done, truncated, info

    def render(self):
        fig = pylab.figure(figsize=[4, 4],  # Inches
                           dpi=200,  # 100 dots per inch, so the resulting buffer is 400x400 pixels
                           )
        fig.gca()
        self.graph.plot_graph()
        width = [self.Footprint[u][v]['Speed'] if self.Footprint[u][v]['Speed'] <= 10 else 10 for u, v in
                 self.Footprint.edges()]
        pos = {n: [n.position[1], n.position[0]] for n in list(self.Footprint.nodes())}
        nx.draw_networkx(self.Footprint, with_labels=False, pos=pos, node_color='red', edge_color='red', width=width,
                         font_color='red')
        pos_for_speed = {n: [n.position[1] - 5, n.position[0] - 5] for n in list(self.Footprint.nodes())}
        nx.draw_networkx_edge_labels(self.Footprint, pos_for_speed,
                                     edge_labels={(u, v): "Speed:{:.0f}".format(d['Speed']) for u, v, d in
                                                  self.Footprint.edges(data=True)}, font_color='red')
        nodes = []
        for node in list(self.Footprint.nodes()):
            state = isinstance(node, Start) or isinstance(node, End)
            if state is False:
                nodes.append(node)
        nx.draw_networkx_labels(self.Footprint, pos_for_speed,
                                labels={n: "Duration:{:.0f}".format(self.Footprint.nodes[n]['Duration']) for n in
                                        nodes},
                                font_color='red',
                                font_size=9,
                                horizontalalignment='center')

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
        self.Footprint = nx.DiGraph()
        self.Footprint.add_node(start)
        nodeattr = self.graph.graph.nodes[start].copy()
        nx.set_node_attributes(self.Footprint, {start: nodeattr})
        self.successorsLst = nx.dfs_successors(G=self.graph.graph, source=start, depth_limit=1)[start]
        # self.action_space = MultiDiscrete([len(self.successorsLst), 100, 10])
        self.action_space = MultiDiscrete([100, 100, 100])
        self.observation_space = Dict({'route': Discrete(100), 'Speed': Discrete(100), 'Duration': Discrete(100),
                                       'pathDistance': Box(low=0.0, high=30000000.0, shape=(1,), dtype=float)})
        self.current_place = start
        self.route_state = {'footprint': self.Footprint, 'route': [start], 'Speed': [], 'Duration': []}
        self.previous_place = None
        self.route_length = 10
        self.state = {'route': locations.index(start), 'Speed': 0, 'Duration': 0, 'pathDistance': 0.0}
        info = self.get_info()
        return self.state, info

    def retrieve_location(self, action):
        location = self.successorsLst[action[0]]
        return

    def update_action_space(self):
        # update action space
        if not isinstance(self.current_place, End):
            successors = nx.dfs_successors(G=self.graph.graph, source=self.current_place, depth_limit=1)
            print('current_place', self.current_place.name)
            self.successorsLst = successors[self.current_place]
            print([loc.name for loc in self.successorsLst])
            self.action_space = MultiDiscrete([len(self.successorsLst), 100, 10])
        else:
            successors = nx.dfs_successors(G=self.graph.graph, source=start, depth_limit=1)
            self.successorsLst = successors[start]
            self.action_space = MultiDiscrete([len(self.successorsLst), 100, 10])

    def action_masks(self):
        if not isinstance(self.current_place, End):
            successors = nx.dfs_successors(G=self.graph.graph, source=self.current_place, depth_limit=1)
            #print('current_place', self.current_place.name)
            self.successorsLst = successors[self.current_place]
            #print([loc.name for loc in self.successorsLst])
            location_mask = [1 if loc in self.successorsLst else 0 for loc in locations]
        else:
            successors = nx.dfs_successors(G=self.graph.graph, source=start, depth_limit=1)
            self.successorsLst = successors[start]
            location_mask = [1 if loc in self.successorsLst else 0 for loc in locations]
        #print('location mask:', location_mask)
        #action_mask = np.array([np.array(location_mask), np.array([1] * 100), np.array([1] * 10)], dtype=object)
        location_mask = np.array(location_mask+[0]*93, dtype=np.int8)
        speed_mask = np.ones((100,), dtype=np.int8)
        duration_mask = [1]*10 + [0]*90
        duration_mask = np.array(duration_mask, dtype=np.int8)
        #action_mask = np.array([location_mask, speed_mask, duration_mask])
        action_mask = (location_mask, speed_mask, duration_mask)
        print(action_mask)
        return action_mask

def mask_fn(env):
    return env.action_masks()
envR = Routing(Graph=graph1, Ship=ship, start=start)
#envR = FlattenObservation(envR)
envR = ActionMasker(envR, mask_fn)

# episodes = 5
# for episode in range(1, episodes +1):
#     done = False
#     obs, _ = envR.reset()
#     score = 0
#     length = 16
#     while not done:
#         if not isinstance(envR.current_place, End):
#             successors = nx.dfs_successors(G=envR.graph.graph, source=envR.current_place, depth_limit=1)
#             #print('current_place', envR.current_place.name)
#             envR.successorsLst = successors[envR.current_place]
#             #print([loc.name for loc in envR.successorsLst])
#             location_mask = [1 if loc in envR.successorsLst else 0 for loc in locations]
#         else:
#             successors = nx.dfs_successors(G=envR.graph.graph, source=start, depth_limit=1)
#             envR.successorsLst = successors[start]
#             location_mask = [1 if loc in envR.successorsLst else 0 for loc in locations]
#         print('location mask:', location_mask)
#         print('successors:', envR.successorsLst)
#         action = envR.action_space.sample(mask=(np.array(location_mask, dtype=np.int8), np.ones((100,), dtype=np.int8), np.ones((10,), dtype=np.int8)))
#         action = envR.action_space.sample()
#         print('action chosen:', action[0])
#         obs, reward, done, info = envR.step(action)
#         envR.render()
#         score += reward
#     print('episode:{} distance:{}'.format(episode, reward))
# envR.close()
"""
episodes = 3
score_list = []
for episode in range(1, episodes +1):
    done = False
    obs, _ = envR.reset()
    score = 0
    length = 10
    while not done:
        pygame.time.delay(100)
        action = envR.action_space.sample(mask=envR.action_masks())
        print('action chosen:', action)
        obs, reward, done, truncated, info = envR.step(action)
        envR.render()
        score += reward
    score_list.append(reward)
    print('episode:{} distance:{}'.format(episode, reward))
print(np.mean(score_list))
print(np.std(score_list))
envR.close()
"""

total_timesteps = 100000

log_path = os.path.join('Training', 'Logs')
model_path = os.path.join('Training', 'Model' f'{total_timesteps}stepModel')
model = MaskablePPO("MultiInputPolicy", envR, verbose=1, tensorboard_log=log_path, seed=100)
model.learn(total_timesteps=total_timesteps)
model.save(model_path)

#envR = FlattenObservation(envR)
model_path = os.path.join('Training', 'Model', f'{total_timesteps}stepModel')
model = MaskablePPO.load(model_path, env=envR)
#eva = evaluate_policy(model, envR, n_eval_episodes=100, render=True)
#print(eva)

vec_env = model.get_env()
obs  = vec_env.reset()
print(obs)
action, _ = model.predict(obs, action_masks=envR.action_masks())
print(action[0])
obs, reward, done, truncated = vec_env.step(action)
print('step output',obs, reward, done, truncated)
episodes = 100
score_list = []
for episode in range(1, episodes + 1):
    done = False
    obs = vec_env.reset()
    print(obs)
    score = 0
    length = 10
    while not done:
        envR.render()
        print('input_obs', obs)
        action, _ = model.predict(obs, action_masks=envR.action_masks()) #, action_masks=envR.action_masks()
        print('action chosen:', action[0])
        obs, reward, done, truncated = vec_env.step(action)
        print('output_obs', obs, reward, done, truncated)
        score += reward
        print(-reward)
    score_list.append(reward)
    print('episode:{} distance:{}'.format(episode, -reward))
print(np.mean(score_list))
print(np.std(score_list))
envR.close()
"""
# envR = FlattenObservation(envR)
# obs,_ = envR.reset()
# print(obs)
# action, _ = model.predict(obs, action_masks=envR.action_masks())
# print(action[0])
# obs, reward, done, truncated, info = envR.step(action)
# print('step output',obs, reward, done, truncated, info)
# episodes = 100
# score_list = []
# for episode in range(1, episodes + 1):
#     done = False
#     obs,_ = envR.reset()
#     print(obs)
#     score = 0
#     length = 10
#     while not done:
#         envR.render()
#         print('input_obs', obs)
#         action, _ = model.predict(obs, action_masks=envR.action_masks()) #, action_masks=envR.action_masks()
#         print('action chosen:', action[0])
#         obs, reward, done, truncated, info = envR.step(action)
#         print('output_obs', obs)
#         score += reward
#     score_list.append(reward)
#     print('episode:{} distance:{}'.format(episode, -reward))
# print(np.mean(score_list))
# print(np.std(score_list))
# envR.close()
"""