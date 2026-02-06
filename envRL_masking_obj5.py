import gym.vector
import gymnasium.spaces.utils
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
from numpy import float64
from pygame.locals import *
import time
import copy
import multiprocessing

import matplotlib.backends.backend_agg as agg

# import stable_baselines3
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# import sb3_contrib
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# render related


# routing related
import networkx as nx
from statistics import mode
import matplotlib
import matplotlib.pyplot as plt
import main
import NSGA
from main import Graph, FishGround, FishFactory, WayPoint, Start, End, Position, great_circle_distance, \
    scgraph_distance, pathRisk, pathCatch, pathGain, pathFuel, pathDistance, Ship, dynamic_routing, \
    passageFuel_Consumption, passageRisk_trial, locFuel_Consumption, locRisk_trial


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

# Create a class function for making training environment
# Objective: Trade catch with route risk

class Routing(Env):
    def __init__(self, Graph: Graph, Ship: Ship, start):
        self.time = 0
        self.ship = copy.deepcopy(Ship)
        self.graph = Graph
        self.Footprint = nx.DiGraph()
        self.Footprint.add_node(start)
        nodeattr = self.graph.graph.nodes[start].copy()
        nx.set_node_attributes(self.Footprint, {start: nodeattr})
        self.successorsLst = nx.dfs_successors(G=self.graph.graph, source=start, depth_limit=1)[start]
        self.action_space = MultiDiscrete([10, 10, 10])
        # old_action_space=Dict({'location_to_go': Discrete(len(self.successorsLst)), 'speed': Box(low=0, high=30, shape=(1,)), 'duration': Box(0, 10, shape=(1,))})
        self.observation_space = Dict({'route': Discrete(10),
                                       'Speed': Discrete(20),
                                       'Duration': Discrete(11),
                                       'pathCatch': Box(low=0, high=1000, shape=(1,), dtype=float),
                                       'pathDistance': Box(low=0, high=30000, shape=(1,), dtype=float),
                                       'pathRisk': Box(low=0, high=30000, shape=(1,), dtype=float)})
        self.current_place = start
        locations.index(start)
        self.state = {'route': locations.index(start), 'Speed': 0, 'Duration': 0, 'pathCatch': 0, 'pathDistance': 0, 'pathRisk': 0}
        self.route_state = {'footprint': self.Footprint, 'route': [start], 'Speed': [], 'Duration': []}
        self.previous_place = None
        self.route_length = 10

    def step(self, action):
        # print(action[0])
        #print('successors:', [loc.name for loc in self.successorsLst])
        location = locations[action[0]]
        # print(location.name)
        self.previous_place = self.current_place
        # Update state, including Footprint, route, speed and duration
        self.current_place = location
        speed = self.get_speed(action[1])
        duration = self.get_duration(action[2])
        self.route_state['route'].append(self.current_place)
        self.route_state['Speed'].append(speed)
        self.route_state['Duration'].append(duration)
        self.Footprint.add_node(self.current_place)
        self.Footprint.add_edge(self.previous_place, self.current_place)
        edgeattr = self.graph.graph.edges[(self.previous_place, self.current_place)].copy()
        edgeattr['speed'] = speed
        edgeattr['duration'] = edgeattr['distance'] / edgeattr['speed'] if edgeattr['speed'] != 0 else np.inf
        nx.set_edge_attributes(self.Footprint, {(self.previous_place, self.current_place): edgeattr})
        nodeattr = self.graph.graph.nodes[self.current_place].copy()
        nodeattr['duration'] = duration

        nx.set_node_attributes(self.Footprint, {self.current_place: nodeattr})
        if isinstance(self.current_place, FishGround):
            fishtype = self.ship.targeted_fish
            newcatch = self.ship.update_catch(self.current_place)
            stock = self.current_place.fishstock
            stock[fishtype]['quantity'] -= newcatch
            self.current_place.update_stock(stock)
            self.graph.update_node_attribute(self.current_place, 'fishstock', stock)
        else:
            newcatch = 0
        distance = self.Footprint.edges[(self.previous_place, self.current_place)]['distance']
        risk = edgeattr['duration'] + nodeattr['duration']*100
        self.state = {'route': action[0], 'Speed': speed, 'Duration': duration, 'pathCatch': self.ship.catch, 'pathDistance': distance, 'pathRisk': risk}
        #print(self.state)
        reward = newcatch-risk
        info = {}

        self.route_length -= 1
        # conditions to end episode
        if self.route_length <= 0 or isinstance(self.current_place, End):
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
        width = [self.Footprint[u][v]['speed'] if self.Footprint[u][v]['speed'] <= 10 else 10 for u, v in
                 self.Footprint.edges()]
        pos = {n: [n.position[1], n.position[0]] for n in list(self.Footprint.nodes())}
        nx.draw_networkx(self.Footprint, with_labels=False, pos=pos, node_color='red', edge_color='red', width=width,
                         font_color='red')
        pos_for_speed = {n: [n.position[1] - 5, n.position[0] - 5] for n in list(self.Footprint.nodes())}
        nx.draw_networkx_edge_labels(self.Footprint, pos_for_speed,
                                     edge_labels={(u, v): "S:{:.0f}".format(d['speed']) for u, v, d in
                                                  self.Footprint.edges(data=True)}, font_color='red')
        nodes = []
        for node in list(self.Footprint.nodes()):
            state = isinstance(node, Start) or isinstance(node, End)
            if state is False:
                nodes.append(node)
        nx.draw_networkx_labels(self.Footprint, pos_for_speed,
                                labels={n: "Du:{:.0f}".format(self.Footprint.nodes[n]['duration']) for n in
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

    def get_speed(self, speed):
        speed = speed + 10
        return speed

    def get_duration(self, duration):
        duration = duration + 1
        return duration

    def get_info(self):
        self.info = {}
        return self.info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        loc1.fishstock['Cod']['quantity'] = 100
        loc2.fishstock['Cod']['quantity'] = 500
        loc3.fishstock['Cod']['quantity'] = 300
        self.ship = copy.deepcopy(ship)
        self.Footprint = nx.DiGraph()
        self.Footprint.add_node(start)
        nodeattr = self.graph.graph.nodes[start].copy()
        nx.set_node_attributes(self.Footprint, {start: nodeattr})
        self.successorsLst = nx.dfs_successors(G=self.graph.graph, source=start, depth_limit=1)[start]
        # self.action_space = MultiDiscrete([len(self.successorsLst), 100, 10])
        self.action_space = MultiDiscrete([10, 10, 10])
        self.observation_space = Dict({'route': Discrete(10),
                                       'Speed': Discrete(20),
                                       'Duration': Discrete(11),
                                       'pathCatch': Box(low=0, high=1000, shape=(1,), dtype=float),
                                       'pathDistance': Box(low=0, high=30000, shape=(1,), dtype=float),
                                       'pathRisk': Box(low=0, high=30000, shape=(1,), dtype=float)})
        self.current_place = start
        locations.index(start)
        self.state = {'route': locations.index(start),
                      'Speed': 0,
                      'Duration': 0,
                      'pathCatch': 0,
                      'pathDistance': 0,
                      'pathRisk': 0}
        self.route_state = {'footprint': self.Footprint, 'route': [start], 'Speed': [], 'Duration': []}
        self.previous_place = None
        self.route_length = 10
        info = self.get_info()
        return self.state, info

    def retrieve_location(self, action):
        location = self.successorsLst[action[0]]
        return

    def update_action_space(self):
        # update action space
        if not isinstance(self.current_place, End):
            successors = nx.dfs_successors(G=self.graph.graph, source=self.current_place, depth_limit=1)
            #print('current_place', self.current_place.name)
            self.successorsLst = successors[self.current_place]
            #print([loc.name for loc in self.successorsLst])
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
            # code below exclude fish ground where its stock is 0.
            locLst = []
            for loc in self.successorsLst:
                if isinstance(loc, FishGround):
                    if loc.fishstock['Cod']['quantity'] <= 0:
                        locLst.append(loc)
            newLst = list(filter(lambda loc: loc not in locLst, self.successorsLst))
            self.successorsLst = newLst
            #print([loc.name for loc in self.successorsLst])
            location_mask = [1 if loc in self.successorsLst else 0 for loc in locations]
        else:
            successors = nx.dfs_successors(G=self.graph.graph, source=start, depth_limit=1)
            self.successorsLst = successors[start]
            location_mask = [1 if loc in self.successorsLst else 0 for loc in locations]
        #print('location mask:', location_mask)
        #action_mask = np.array([np.array(location_mask), np.array([1] * 100), np.array([1] * 10)], dtype=object)
        #action_mask = np.array([location_mask, speed_mask, duration_mask])
        location_mask = np.array(location_mask + [0] * 3, dtype=np.int8)
        speed_mask = np.ones((10,), dtype=np.int8)
        duration_mask = np.ones((10,), dtype=np.int8)
        duration_mask = np.array(duration_mask, dtype=np.int8)
        # action_mask = np.array([location_mask, speed_mask, duration_mask])
        action_mask = (location_mask, speed_mask, duration_mask)
        #print(action_mask)
        return action_mask

def mask_fn(env):
    return env.action_masks()
envR = Routing(Graph=graph1, Ship=ship, start=start)
envR = ActionMasker(envR, mask_fn)

