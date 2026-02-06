"""Generate fishing route. To generate route, several functions will be needed. How to structure all the functions?
Multistage:
    Stage 1: Whole voyage planning Generate many paths first and evaluate all of them based on objectives
    Stage 2: 3 days horizon planning, due to the trust on 3 days weather forcast
    Stage 3: 1 hours horizon planning
To plan the voyage, way points need to be generated. passages can be generated from way points. attributes can be
assigned to all the generated way points and passages.
"""
import time
from statistics import mode

import geopy.distance
import networkx as nx
import numpy as np
import pandas
from matplotlib import pyplot as plt

from typing import NamedTuple
import searoute
from scgraph.geographs.marnet import marnet_geograph
from matplotlib import animation
from matplotlib.animation import PillowWriter
import NSGA

class Grid:
    """Grid created from map"""
    def generategrid(self, map, resolution):
        grid = map + resolution
        return grid


class Obstacle:
    """Obstacles which is defined by a list of points"""
    def __init__(self, obstacle):
        self.obstacle = obstacle

    def boundary_to_avoid_allision(self, dcpa):
        """
        :param obstacle: edge points of obstacle
        :param dcpa: distance to the closest point of approach. this is the safe distance between ship and obstacles
        :return: safe boundary which is defined by a list of points
        """
        waypoint1 = self.obstacle + dcpa
        waypoint2 = self.obstacle + dcpa
        return waypoint1, waypoint2


class Routes:
    """Routes: a network contains all possible routes"""
    def __init__(self, network, start, end):
        self.graph = network
        self.shortest_path = list(nx.shortest_simple_paths(self.graph, start, end))

    def update_routes(self, network, start, end):
        self.graph = network
        self.shortest_path = list(nx.shortest_simple_paths(self.graph, start, end))


class Map:
    """Map object is created for maps, i.e., seabed terrain, land"""

    def __init__(self, map):
        self.map = map

class Position(NamedTuple):
    lat: float
    long: float

class Graph:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.name = 'graph1'

    def update_node_attribute(self, node, attribute, newvalue):
        self.graph.nodes[node][attribute] = newvalue

    def updage_edge_attribute(self, edge, attribute, newvalue):
        self.graph.edges[edge][attribute] = newvalue

    def addlocations_2(self, locations):
        """
        :param locations: locations should be a list [loc1, loc2, loc3, loc4] each location is a class object which has certain properties.
        :return: a new graph with added locations and edges.
        """
        oldgraph = self.graph.copy()
        old_nodes = {'all': oldgraph.nodes}
        print(old_nodes)
        old_nodes['fish ground'] = [val for val in old_nodes['all'] if
                                    isinstance(val, FishGround)]
        old_nodes['fish factory'] = [val for val in old_nodes['all'] if
                                     isinstance(val, FishFactory)]
        old_nodes['way point'] = [val for val in old_nodes['all'] if
                                  isinstance(val, WayPoint)]
        old_nodes['start'] = [val for val in old_nodes['all'] if isinstance(val, Start)]
        old_nodes['end'] = [val for val in old_nodes['all'] if isinstance(val, End)]
        for loc in locations:
            name = loc.name
            self.graph.add_node(loc)
            node = self.graph.nodes[loc]
            locproperties = vars(loc)
            for key in locproperties:
                node[key] = locproperties[key]
                print(node[key])

        nodes = self.graph.nodes
        res = [i for i in nodes if i not in old_nodes['all']]
        new_nodes = {'all': res}
        new_nodes['fish ground'] = [val for val in new_nodes['all'] if
                                    isinstance(val, FishGround)]
        new_nodes['fish factory'] = [val for val in new_nodes['all'] if
                                     isinstance(val, FishFactory)]
        new_nodes['way point'] = [val for val in new_nodes['all'] if
                                  isinstance(val, WayPoint)]
        new_nodes['start'] = [val for val in new_nodes['all'] if isinstance(val, Start)]
        new_nodes['end'] = [val for val in new_nodes['all'] if isinstance(val, End)]
        print(new_nodes)
        for old_node in old_nodes['fish ground']:
            for new_node in new_nodes['fish ground']:
                dis = great_circle_distance(self.graph.nodes[old_node]['position'], self.graph.nodes[new_node]['position'])
                self.graph.add_edge(old_node, new_node, distance=dis)
                self.graph.add_edge(new_node, old_node, distance=dis)
            for new_node in new_nodes['fish factory']:
                self.graph.add_edge(old_node, new_node, distance=great_circle_distance(self.graph.nodes[old_node]['position'],
                                                                                       self.graph.nodes[new_node]['position']))
            for new_node in new_nodes['start']:
                self.graph.add_edge(new_node, old_node, distance=great_circle_distance(self.graph.nodes[old_node]['position'],
                                                                                       self.graph.nodes[new_node]['position']))
        for new_node_p in new_nodes['fish ground']:
            new_nodes_copy = new_nodes['fish ground'].copy()
            new_nodes_copy.remove(new_node_p)
            for new_node_q in new_nodes_copy:
                dis = great_circle_distance(self.graph.nodes[new_node_p]['position'], self.graph.nodes[new_node_q]['position'])
                self.graph.add_edge(new_node_p, new_node_q, distance=dis)
                self.graph.add_edge(new_node_q, new_node_p, distance=dis)
                print(new_node_p.name, new_node_q.name, dis)
            for new_node_q in new_nodes['fish factory']:
                dis = great_circle_distance(self.graph.nodes[new_node_p]['position'], self.graph.nodes[new_node_q]['position'])
                self.graph.add_edge(new_node_p, new_node_q, distance=dis)
                print(new_node_p.name, new_node_q.name, dis)
            for new_node_q in new_nodes['start']:
                dis = great_circle_distance(self.graph.nodes[new_node_p]['position'], self.graph.nodes[new_node_q]['position'])
                self.graph.add_edge(new_node_q, new_node_p, distance=dis)
        for new_node_p in new_nodes['fish factory']:
            for new_node_q in new_nodes['end']:
                print(self.graph.nodes[new_node_p]['position'], self.graph.nodes[new_node_q]['position'])
                dis = great_circle_distance(self.graph.nodes[new_node_p]['position'], self.graph.nodes[new_node_q]['position'])
                self.graph.add_edge(new_node_p, new_node_q, distance=dis)
                print(new_node_p.name, new_node_q.name, dis)

    def addlocations(self, locations):
        """
        :param locations: locations should be a list [loc1, loc2, loc3, loc4] each location is a class object which has certain properties.
        :return: a new graph with added locations and edges.
        """
        oldgraph = self.graph.copy()
        old_nodes = {'all': oldgraph.nodes}
        print(old_nodes)
        old_nodes['fish ground'] = [val for val in old_nodes['all'] if isinstance(self.graph.nodes[val]['object'], FishGround)]
        old_nodes['fish factory'] = [val for val in old_nodes['all'] if isinstance(self.graph.nodes[val]['object'], FishFactory)]
        old_nodes['way point'] = [val for val in old_nodes['all'] if isinstance(self.graph.nodes[val]['object'], WayPoint)]
        old_nodes['start'] = [val for val in old_nodes['all'] if isinstance(self.graph.nodes[val]['object'], Start)]
        old_nodes['end'] = [val for val in old_nodes['all'] if isinstance(self.graph.nodes[val]['object'], End)]
        for loc in locations:
            name = loc.name
            self.graph.add_node(name)
            self.graph.nodes[name]['object'] = loc
            # self.graph.nodes[name]['object'] = loc
            # print(self.graph.nodes[name]['object'].position[0])
            locproperties = vars(loc)
            for key in locproperties:
                if key != 'name':
                    self.graph.nodes[name][key] = locproperties[key]
                    print(self.graph.nodes[name][key])

        nodes = self.graph.nodes
        print(old_nodes['all'])
        res = [i for i in nodes if i not in old_nodes['all']]
        new_nodes = {'all': res}

        new_nodes['fish ground'] = [val for val in new_nodes['all'] if isinstance(self.graph.nodes[val]['object'], FishGround)]
        new_nodes['fish factory'] = [val for val in new_nodes['all'] if isinstance(self.graph.nodes[val]['object'], FishFactory)]
        new_nodes['way point'] = [val for val in new_nodes['all'] if isinstance(self.graph.nodes[val]['object'], WayPoint)]
        new_nodes['start'] = [val for val in new_nodes['all'] if isinstance(self.graph.nodes[val]['object'], Start)]
        new_nodes['end'] = [val for val in new_nodes['all'] if isinstance(self.graph.nodes[val]['object'], End)]
        print(new_nodes)
        for old_node in old_nodes['fish ground']:
            for new_node in new_nodes['fish ground']:
                dis = great_circle_distance(self.graph.nodes[old_node]['position'], self.graph.nodes[new_node]['position'])
                self.graph.add_edge(old_node, new_node, distance=dis)
                self.graph.add_edge(new_node, old_node, distance=dis)
            for new_node in new_nodes['fish factory']:
                self.graph.add_edge(old_node, new_node, distance=great_circle_distance(self.graph.nodes[old_node]['position'], self.graph.nodes[new_node]['position']))
            for new_node in new_nodes['start']:
                self.graph.add_edge(new_node, old_node, distance=great_circle_distance(self.graph.nodes[old_node]['position'], self.graph.nodes[new_node]['position']))

        for new_node_p in new_nodes['fish ground']:
            new_nodes_copy = new_nodes['fish ground'].copy()
            new_nodes_copy.remove(new_node_p)
            for new_node_q in new_nodes_copy:
                dis = great_circle_distance(self.graph.nodes[new_node_p]['position'], self.graph.nodes[new_node_q]['position'])
                self.graph.add_edge(new_node_p, new_node_q, distance=dis)
                self.graph.add_edge(new_node_q, new_node_p, distance=dis)
            for new_node_q in new_nodes['fish factory']:
                dis = great_circle_distance(self.graph.nodes[new_node_p]['position'], self.graph.nodes[new_node_q]['position'])
                self.graph.add_edge(new_node_p, new_node_q, distance=dis)
            for new_node_q in new_nodes['start']:
                dis = great_circle_distance(self.graph.nodes[new_node_p]['position'], self.graph.nodes[new_node_q]['position'])
                self.graph.add_edge(new_node_q, new_node_p, distance=dis)
        for new_node_p in new_nodes['fish factory']:
            for new_node_q in new_nodes['end']:
                dis = great_circle_distance(self.graph.nodes[new_node_p]['position'], self.graph.nodes[new_node_q]['position'])
                self.graph.add_edge(new_node_p, new_node_q, distance=dis)

    def add_Waypoint_from_waypoints(self, predecessor, waypoints, successor):

        dictionary = {waypoint: [vars(waypoint)] for waypoint in waypoints}
        self.graph.add_nodes_from(dictionary)
        newWPObjLst = [predecessor] + waypoints + [successor]
        for i in range(0, len(newWPObjLst)-1):
            dis = great_circle_distance(newWPObjLst[i].position, newWPObjLst[i+1].position)
            self.graph.add_edge(newWPObjLst[i], newWPObjLst[i+1], distance=dis)
        self.graph.remove_edge(predecessor, successor)

    def addWaypoint_from_coordinates(self, predecessor, waypoints, successor):
        """
        :param predecessor: node before waypoints
        :param waypoints:
        :param successor: next node that the last waypoints direct to.
        :return:
        """
        WPObjLst = [WayPoint(name='WP%s' % i, position=Position(waypoint[0], waypoint[1])) for (i, waypoint) in zip(range(1, len(waypoints)+1), waypoints)]
        dictionary = {waypoint: [vars(waypoint)] for waypoint in WPObjLst}
        self.graph.add_nodes_from(dictionary)
        newWPObjLst = [predecessor] + WPObjLst + [successor]
        for i in range(0, len(newWPObjLst)-1):
            dis = great_circle_distance(newWPObjLst[i].position, newWPObjLst[i+1].position)
            self.graph.add_edge(newWPObjLst[i], newWPObjLst[i+1], distance=dis)
        self.graph.remove_edge(predecessor, successor)


    def addedges(self, edges):
        """
        To add edges.
        :param edges: [(loc1, loc2), (loc2, loc3), (loc3, loc4)...],{'edge1':{'edge':(loc1, loc2), 'risk': 100, 'distance': 300, 'weather condition':...}}
        :return:
        """

    def addedge(self, node_p, node_q, direction=None):

        """
        add a directional edge between two node_p and node_q. node_p is the starting node and node_q is the destination by default.
        :param node_p: starting node
        :param node_q: destination
        :param direction: whether it is "bothways" or (node_p, node_q) or (node_q, node_p) or no specification.
        :return:
        """
        if direction == 'bothways':
            dis = great_circle_distance(self.graph.node[node_p]['position'], self.graph.node[node_q]['position'])
            self.graph.add_edge(node_p, node_q, distance=dis)
            self.graph.add_edge(node_q, node_p, distance=dis)
        else:
            dis = great_circle_distance(self.graph.node[node_p]['position'], self.graph.node[node_q]['position'])
            self.graph.add_edge(node_p, node_q, distance=dis)

    def plot_graph(self):
        """
        Plot graph according to its coordinates.
        :return:
        """
        pos = {n: [n.position[1], n.position[0]] for n in list(self.graph.nodes)}
        nx.draw_networkx_nodes(self.graph, pos)
        nx.draw_networkx_labels(self.graph, pos, labels={n: n.name for n in list(self.graph.nodes)})
        fishground = list(filter(lambda node: isinstance(node, FishGround), list(self.graph.nodes())))
        stockPos = {n: [n.position[1]+3, n.position[0]+3] for n in list(fishground)}
        nx.draw_networkx_labels(self.graph, stockPos,
                                labels={n: 'Fishstock: {}'.format(n.fishstock['Cod']['quantity']) for n in fishground},
                                font_color='green')
        # nx.draw_networkx_edge_labels(self.graph, pos, edge_labels={(u, v): "D:{:.0f}".format(d['distance']) for u, v, d in
        #                                                   self.graph.edges(data=True)})

    def simple_routes(self, start, end):
        self.routes = list(nx.shortest_simple_paths(self.graph, start, end))


    def routes_graph(self, start, end):
        routes_graph = nx.DiGraph()
        routes = list(nx.shortest_simple_paths(self.graph, start, end))
        new_start = routes[0][0].name
        new_end = routes[0][-1].name
        for i, route in zip(range(0, len(routes)), routes):
            k = 1
            pre_node_attr = self.graph.nodes[route[0]].copy()
            pre_node_attr['object'] = route[0]
            pre_node = route[0].name
            routes_graph.add_node(pre_node)
            nx.set_node_attributes(routes_graph, {pre_node: pre_node_attr})
            while k < (len(route) - 1):
                new_node_attr = self.graph.nodes[route[k]].copy()
                new_node = 'R' + str(i) + '-' + route[k].name
                edge_attr = self.graph.edges[(route[k - 1], route[k])].copy()
                new_edge = {(pre_node, new_node): edge_attr}
                new_node_attr['object'] = route[k]
                routes_graph.add_node(new_node)
                nx.set_node_attributes(routes_graph, {new_node: new_node_attr})
                routes_graph.add_edge(pre_node, new_node)
                nx.set_edge_attributes(routes_graph, new_edge)
                pre_node = new_node
                k += 1
            new_node_attr = self.graph.nodes[route[-1]].copy()
            new_node_attr['object'] = route[-1]
            new_node = route[-1].name
            routes_graph.add_node(new_node)
            nx.set_node_attributes(routes_graph, {new_node: new_node_attr})
            edge_attr = self.graph.edges[(route[-2], route[-1])].copy()
            new_edge = {(pre_node, new_node): edge_attr}
            routes_graph.add_edge(pre_node, new_node)
            nx.set_edge_attributes(routes_graph, new_edge)
        routes = list(nx.shortest_simple_paths(routes_graph, new_start, new_end))
        return routes_graph, routes

    def generate_speed_matrix(self, routes_graph, routes):
        routes_graph = routes_graph
        routes = routes
        speed_matrix = [[20]*(len(routes[i])-1) for i in range(0, len(routes))]
        return speed_matrix


    def generate_fishing_duration_matrix(self, routes_graph, routes):
        routes_graph = routes_graph
        routes = routes
        node_duration_matrix = [[20]*(len(routes[i])-2) for i in range(0, len(routes))]
        return node_duration_matrix


    def ship_timing_speed_specific_graph(self, ship, hazards, node_duration_matrix, speed_matrix, routes_graph, routes):
        """
            This function is to create a new routes map with updated speed, risk, catch information.
        :param ship:
        :param node_duration_matrix:
        :param speed_matrix:
        :param routes_graph:
        :param routes:
        :param start:
        :param end:
        :return:
        """
        ship = ship
        NewGraph = routes_graph
        "update_attributes(Graph)"
        routes = routes
        for j, route in zip(range(0, len(routes)), routes):
            time = 0
            pre_node = route[0]
            k = 1  # k is node index in a route
            while k < (len(route)-1):
                current_node = route[k]
                """Code below is to assign edge attribute"""
                edge_attr = NewGraph.edges[(pre_node, current_node)].copy()
                edge_attr['speed'] = speed_matrix[j][k-1]
                edge_attr['duration'] = edge_attr['distance']/edge_attr['speed']
                edge_attr['starting time'] = time
                edge_attr['finishing time'] = time + edge_attr['duration']
                """
                edge_attr['risk'] = passageRisk(startLoc=NewGraph.nodes[pre_node]['position'],
                                                endLoc=NewGraph.nodes[current_node]['position'],
                                                startTime=edge_attr['starting time'],
                                                endTime=edge_attr['finishing time'],
                                                hazards=hazards,
                                                ship=ship)
                
                edge_attr['fuelconcumption'] = fuel_Consumption(ship,
                                                                edge_attr['distance'],
                                                                edge_attr['speed'])
                """
                updated_edge = {(pre_node, current_node): edge_attr}
                nx.set_edge_attributes(NewGraph, updated_edge)
                """Code below is to assign node attribute"""
                current_node_attr = NewGraph.nodes[current_node].copy()
                current_node_attr['duration'] = node_duration_matrix[j][k-1]
                current_node_attr['arrival time'] = time + edge_attr['duration']
                current_node_attr['departure time'] = current_node_attr['arrival time'] + current_node_attr['duration']
                nx.set_node_attributes(NewGraph, {current_node: current_node_attr})
                """Code below is for the next node"""
                pre_node = current_node
                time += edge_attr['duration'] + current_node_attr['duration']
                k += 1
            """Code below is to assign edge attribute to the last edge of the route"""
            current_node = route[-1]
            edge_attr = NewGraph.edges[(pre_node, current_node)].copy()
            edge_attr['speed'] = speed_matrix[j][k-1]
            edge_attr['duration'] = edge_attr['distance'] / edge_attr['speed']
            edge_attr['starting time'] = time
            edge_attr['finishing time'] = time + edge_attr['duration']
            current_node_attr = NewGraph.nodes[current_node].copy()
            current_node_attr['arrival time'] = edge_attr['finishing time']
            nx.set_node_attributes(NewGraph, {current_node: current_node_attr})
            updated_edge = {(pre_node, current_node): edge_attr}
            nx.set_edge_attributes(NewGraph, updated_edge)
            time += edge_attr['duration']
        return NewGraph

    def ship_timing_speed_specific_route(self, ship, speedLst, durationLst, routes_graph, route):
        ship = ship
        "create a subgraph"
        NewGraph = routes_graph.subgraph(route).copy()
        route = route
        time = 0
        pre_node = route[0]
        k = 1  # k is node index in a route
        while k < (len(route)-1):
            current_node = route[k]
            """Code below is to assign edge attribute"""
            edge_attr = NewGraph.edges[(pre_node, current_node)].copy()
            edge_attr['speed'] = speedLst[k - 1]
            edge_attr['duration'] = edge_attr['distance'] / edge_attr['speed'] if edge_attr['speed'] != 0 else np.inf
            edge_attr['starting time'] = time
            edge_attr['finishing time'] = time + edge_attr['duration']
            edge_attr['risk'] = passageRisk_trial(ship, speed=edge_attr['speed'], distance=edge_attr['distance'])
            edge_attr['fuel consumption'] = passageFuel_Consumption(ship,
                                                                   edge_attr['distance'],
                                                                   edge_attr['speed'])
            updated_edge = {(pre_node, current_node): edge_attr}
            nx.set_edge_attributes(NewGraph, updated_edge)
            """Code below is to assign node attribute"""
            current_node_attr = NewGraph.nodes[current_node].copy()
            current_node_attr['duration'] = durationLst[k - 1]
            current_node_attr['arrival time'] = time + edge_attr['duration']
            current_node_attr['departure time'] = current_node_attr['arrival time'] + current_node_attr['duration']
            current_node_attr['fuel consumption'] = locFuel_Consumption(ship, duration=current_node_attr['duration'])
            current_node_attr['risk'] = locRisk_trial(ship, duration=current_node_attr['duration'])
            nx.set_node_attributes(NewGraph, {current_node: current_node_attr})
            """Code below is for the next node"""
            pre_node = current_node
            time += edge_attr['duration'] + current_node_attr['duration']
            k += 1
            """Code below is to assign edge attribute to the last edge of the route"""
        current_node = route[-1]
        edge_attr = NewGraph.edges[(pre_node, current_node)].copy()
        edge_attr['speed'] = speedLst[k - 1]
        edge_attr['duration'] = edge_attr['distance'] / edge_attr['speed'] if edge_attr['speed'] != 0 else np.inf
        edge_attr['starting time'] = time
        edge_attr['finishing time'] = time + edge_attr['duration']
        edge_attr['fuel consumption'] = passageFuel_Consumption(ship,
                                                               edge_attr['distance'],
                                                               edge_attr['speed'])
        edge_attr['risk'] = passageRisk_trial(ship, speed=edge_attr['speed'], distance=edge_attr['distance'])
        current_node_attr = NewGraph.nodes[current_node].copy()
        current_node_attr['arrival time'] = edge_attr['finishing time']
        nx.set_node_attributes(NewGraph, {current_node: current_node_attr})
        updated_edge = {(pre_node, current_node): edge_attr}
        nx.set_edge_attributes(NewGraph, updated_edge)
        time += edge_attr['duration']
        return NewGraph

    def plot_optimals(self, ship, optimized_routes, routes_graph, num):
        """
        The function is to plot all optimized routes in a graph.
        size of the node represent the stop duration at each node.
        Width of line represents the speed.
        Arrival and departure time show on each node.
        :param optimized_routes: routes with nodes, edge speed and node duration.
        :param graph: Graph
        :return: A graph map
        """
        colorlist = ['blue', 'orange', 'green', 'purple', 'brown', 'black', 'grey', 'olive',
                     'cornflowerblue'] * round(1 + num / 9)
        num = len(optimized_routes)
        for i, route in zip(range(0, num), optimized_routes[0:num]):
            print(route)
            graph = self.ship_timing_speed_specific_route(ship=ship, speedLst=route[0][0], durationLst=route[0][1], routes_graph=routes_graph, route=route[1])
            #sposition = [[graph.nodes['S']['position'][1], graph.nodes['S']['position'][0]]]
            #eposition = [[graph.nodes['E']['position'][1], graph.nodes['E']['position'][0]]]
            #others = [[graph.nodes[n]['position'][1], graph.nodes[n]['position'][0]] for n in route[1][1:-1]]
            #allpositions = sposition + others + eposition
            allpositions = [[graph.nodes[n]['position'][1], graph.nodes[n]['position'][0]] for n in route[1]]
            nodes = route[1]
            print('nodes:', nodes)
            pos = {n: allpositions[nodes.index(n)] for n in nodes}
            print(pos)
            rad = (i - num / 2) * 0.05
            edgelist = [(route[1][j], route[1][j + 1]) for j in range(0, len(route[1]) - 1)]
            # edge_labels = {
            #     (u, v): "%s %s" % ("S:{:.0f}".format(d['starting time']), "E:{:.0f}".format(d['finishing time'])) for
            #     u, v, d in
            #     graph.edges(data=True)}
            #print(edge_labels)
            nx.draw(graph, pos, labels={n: graph.nodes[n]['object'].name for n in list(graph.nodes)}, connectionstyle=f'arc3, rad = {rad}', edge_color=colorlist[i])
            #edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
            #print(edge_pos)
            #nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
            #nx.draw_networkx_edges(graph, pos)
        #plt.show()

    def plot_optimals_multigraph(self, optimized_routes):
        num = len(optimized_routes)
        colorlist = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'black', 'grey', 'olive',
                     'cornflowerblue'] * round(1 + num / 10)
        for i, route in zip(range(0, num), optimized_routes[0:num]):
            rad = (i - num / 2) * 0.05
            edgelist = [(route[1][j], route[1][j + 1]) for j in range(0, len(route[1]) - 1)]
            pos = {n:[n.position[1], n.position[0]] for n in list(self.graph.nodes())}
            nx.draw_networkx_edges(self.graph, pos=pos, edgelist=edgelist, connectionstyle=f'arc3, rad = {rad}', edge_color=colorlist[i])

class Weather:
    """
    Weather should include weather parameters, location and time.
    """
    def __init__(self, weather):
        self.weather = weather

    def updateweather(self, newweather):
        self.weather = newweather

    def retrieve_point_weather(self, Weather_map, position, time):
        locweather = Weather_map(position, time)

        return locweather


class Ship:
    """Ship class"""
    def __init__(self, targeted_fish, targeted_quantity, weight):
        self.targeted_fish = targeted_fish
        self.targeted_quantity = targeted_quantity
        self.weight = weight
        self.catch = 0
        self.remaining_quantity = targeted_quantity

    def update_catch(self, fishground):
        difference = self.targeted_quantity - self.catch
        fishtype = self.targeted_fish
        fishstock = fishground.fishstock
        try:
            maximum_catch = fishstock[fishtype]['quantity']
            if (isinstance(maximum_catch, int) or isinstance(maximum_catch, float)) is False:
                maximum_catch = maximum_catch.mean()
        except KeyError:
            maximum_catch = 0
        if difference <= maximum_catch:
            newcatch = difference
            self.catch += difference
        else:
            newcatch = maximum_catch
            self.catch += newcatch
        self.remaining_quantity = self.targeted_quantity - self.catch
        return newcatch



class ShipIceLoad:
    """a class made to represent the amount of ice on vessel."""
    def __init__(self, iceload):
        self.iceload = iceload

    #def update_iceload(self, icerate, duration):
        #self.iceload = Icing.ice_thickness(duration,)

    def update_static_stability(self, ship, iceload):
        static_stability = ship + iceload
        return static_stability


class ShipRisk:
    """A class made to represent the risk that the vessel will take."""
    def __init__(self, risk, ship):
        self.risk = risk


class ShipCatch:
    """A class made to represent the fish the vessel caught."""
    def __init__(self, targeted_fish, catch, ship):
        self.catch = catch

    def empty_catch(self):
        self.catch = 0

    def update_catch(self, node, fishing_time_requirement, targeted_fish, targeted_quantity):
        FishGround = node['object']
        fishstock = FishGround.fishstock[targeted_fish]
        quantity = fishstock['quantity']
        timewindow = fishstock['timewindow']
        if (node['departure time'] <= timewindow[1] & fishing_time_requirement <= node['duration']):
            self.catch += min(targeted_quantity, quantity)


class WayPoint:
    """WayPoint to follow in a path"""
    def __init__(self, name, position):
        self.timewindow = None
        self.name = name
        self.position = position
        
    def set_timewindow(self, timewindow):
        self.timewindow = timewindow


class Start:
    def __init__(self, name, location):
        self.name=name
        self.position = location


class End:
    def __init__(self, name, location):
        self.name = name
        self.position = location


class FishGround:
    """Fish location. fishstock is a dictionary of different type of fish and the amount of fish and timewindow that fish is there
    fishstock = {'Simon': {'quantity': 100, 'timewindow': (2023/01/01 10am, 2023/01/02 9pm)},
                'Codl':{'quantity': 100, 'timewindow': (2023/01/01 10am, 2023/01/02 9pm)}
                }
    """
    def __init__(self, name, location:Position, fishstock: dict):
        self.name = name
        self.position = location
        self.fishstock = fishstock

    def update_stock(self, fishstock):
        self.fishstock = fishstock

    def update_risk(self, ship, Weather_map, startTime, endTime, hazards):
        self.risk = locRisk(location=self.position, startTime=startTime, endTime=endTime, hazards=hazards)


class FishFactory:
    """Fish factory where fish are sold to from fisherman. The destination"""
    def __init__(self, name, location:Position):
        self.name = name
        self.position = location


class CostFunction:
    """Gain, here cost function is writen"""
    def __init__(self, objectives):
        self.objectives = objectives

    def update_values(self, new_values):
        self.values = new_values


class InformationCollector:
    """To collect information, there is a cost associated with it."""
    def __init__(self, variables_tobe_collected, frequencies):
        self.variables = variables_tobe_collected
        self.frequencies = frequencies
        self.pairs = zip(self.variables, self.frequencies)



class RouteOptimizer:
    "Generate optimized route."

    def opportunistic_route_changer(self, start, target, new_route):
        self.bestroute = new_route

    def optimal_route_threshold(self, start, target, route, threshold):
        self.bestroute = route
        return route

    def optimal_ship_speed_duration_all_routes(self, ship, Graph, start, end, gene_space, objectives, constraints, generation_limits=50, num=10):
        """

        :param ship:
        :param Graph:
        :param start:
        :param end:
        :param gene_space:
        :param objectives:
        :param constraints:
        :param generation_limits:
        :param num:
        :return:
        """
        routes_graph = Graph.routes_graph(start, end)
        newgraph = routes_graph[0]
        routes = routes_graph[1]
        optimals = dict()
        timedict = dict()
        nipy = plt.get_cmap('nipy_spectral')
        for i in range(0, len(routes)):
            route = routes[i]
            initial_solutions = NSGA.generate_initial_solutions(gene_space=gene_space, route=route, num=5)
            #print('initial solutions:', initial_solutions)
            newPathRisk = lambda solution: pathRisk(G=Graph.ship_timing_speed_specific_route(ship=0,
                                                                                              speedLst=solution[0],
                                                                                              durationLst=solution[1],
                                                                                              routes_graph=newgraph,
                                                                                              route=route),
                                                    path=route)
            newPathDistance = lambda solution: pathDistance(G=Graph.ship_timing_speed_specific_route(ship=0,
                                                                                              speedLst=solution[0],
                                                                                              durationLst=solution[1],
                                                                                              routes_graph=newgraph,
                                                                                              route=route),
                                                    path=route)

            newPathFuel = lambda solution: pathFuel(G=Graph.ship_timing_speed_specific_route(ship=0,
                                                                                              speedLst=solution[0],
                                                                                              durationLst=solution[1],
                                                                                              routes_graph=newgraph,
                                                                                              route=route),
                                                    path=route)

            newPathCatch = lambda solution: pathCatch(G=Graph.ship_timing_speed_specific_route(ship=0,
                                                                                              speedLst=solution[0],
                                                                                              durationLst=solution[1],
                                                                                              routes_graph=newgraph,
                                                                                              route=route),
                                                      ship=ship,
                                                      path=route)
            objectivefuncs = {'minimize': [], 'maximize': []}
            constraint_funcs = []
            for objective in objectives['minimize']:
                if 'risk' in objective or 'Risk' in objective:
                    objectivefuncs['minimize'].append(newPathRisk)
                if 'distance' in objective or 'Distance' in objective:
                    objectivefuncs['minimize'].append(newPathDistance)
                if 'fuel' in objective or 'Fuel' in objective:
                    objectivefuncs['minimize'].append(newPathFuel)
                if 'catch' in objective or 'Catch' in objective:
                    objectivefuncs['minimize'].append(newPathCatch)
            for objective in objectives['maximize']:
                if 'risk' in objective or 'Risk' in objective:
                    objectivefuncs['maximize'].append(newPathRisk)
                if 'distance' in objective or 'Distance' in objective:
                    objectivefuncs['maximize'].append(newPathDistance)
                if 'fuel' in objective or 'Fuel' in objective:
                    objectivefuncs['maximize'].append(newPathFuel)
                if 'catch' in objective or 'Catch' in objective:
                    objectivefuncs['maximize'].append(newPathCatch)
            try:
                for const in list(constraints.keys()):
                    if 'risk' in const or 'Risk' in const:
                        constraint_funcs.append(newPathRisk)
                    if 'distance' in const or 'Distance' in const:
                        constraint_funcs.append(newPathDistance)
                    if 'fuel' in const or 'Fuel' in const:
                        constraint_funcs.append(newPathFuel)
                    if 'catch' in const or 'Catch' in const:
                        constraint_funcs.append(newPathCatch)
            except KeyError:
                constraint_funcs = []
                constraints = {}

            #newPathCatch
            starttime = time.time()
            optimal_solutions = NSGA.run_evolution(initial_solutions=initial_solutions, gene_space=gene_space,
                                                   objective_funcs=objectivefuncs, constraint_funcs=constraint_funcs, constraints=constraints,
                                                   generation_limit=generation_limits, num=num)
            endtime = time.time()
            #print('route %s' % route)
            #print('optimal speed and duration %s' % optimal_solutions)
            #print('time used for finding optimal speed and duration:', endtime-starttime)
            optimals['%s' % route] = optimal_solutions
        #     timedict['%s' % route] = endtime-starttime
        #     risk = [objectivefuncs['minimize'][0](optimal_solutions[i]) for i in range(0, len(optimal_solutions))]
        #     fuel = [objectivefuncs['maximize'][0](optimal_solutions[i]) for i in range(0, len(optimal_solutions))]
        #     plt.scatter(risk, fuel, color=nipy(i / 100), marker="s", label='Route: %s' % i)
        # plt.legend(fontsize=5)
        # plt.show()
        # plt.plot(timedict.keys(), timedict.values())
        # plt.show()
        return optimals

    def get_optimal_route_from_all_routes(self, ship, Graph, start, end, gene_space, objectives, constraints, generation_limits=50, num=10):
        routes_graph = Graph.routes_graph(start, end)
        newgraph = routes_graph[0]
        routes = routes_graph[1]
        tourments = self.optimal_ship_speed_duration_all_routes(ship, Graph, start, end, gene_space, objectives, constraints, generation_limits=generation_limits, num=num)
        #print(tourments)
        #totalroutes= sum([len(tourments[key]) for key in list(tourments.keys())])
        newPathRisk = lambda solution: pathRisk(G=Graph.ship_timing_speed_specific_route(ship=0,
                                                                                         speedLst=solution[0][0],
                                                                                         durationLst=solution[0][1],
                                                                                         routes_graph=newgraph,
                                                                                         route=solution[1]),
                                                path=solution[1])
        newPathDistance = lambda solution: pathDistance(G=Graph.ship_timing_speed_specific_route(ship=0,
                                                                                         speedLst=solution[0][0],
                                                                                         durationLst=solution[0][1],
                                                                                         routes_graph=newgraph,
                                                                                         route=solution[1]),
                                                path=solution[1])

        newPathFuel = lambda solution: pathFuel(G=Graph.ship_timing_speed_specific_route(ship=0,
                                                                                         speedLst=solution[0][0],
                                                                                         durationLst=solution[0][1],
                                                                                         routes_graph=newgraph,
                                                                                         route=solution[1]),
                                                path=solution[1])
        newPathCatch = lambda solution: pathCatch(G=Graph.ship_timing_speed_specific_route(ship=0,
                                                                                         speedLst=solution[0][0],
                                                                                         durationLst=solution[0][1],
                                                                                         routes_graph=newgraph,
                                                                                         route=solution[1]),
                                                  ship=ship,
                                                  path=solution[1])
        objectivefuncs = {'minimize': [], 'maximize': []}
        constraint_funcs = []
        for objective in objectives['minimize']:
            if 'risk' in objective or 'Risk' in objective:
                objectivefuncs['minimize'].append(newPathRisk)
            if 'distance' in objective or 'Distance' in objective:
                objectivefuncs['minimize'].append(newPathDistance)
            if 'fuel' in objective or 'Fuel' in objective:
                objectivefuncs['minimize'].append(newPathFuel)
            if 'catch' in objective or 'Catch' in objective:
                objectivefuncs['minimize'].append(newPathCatch)
        for objective in objectives['maximize']:
            if 'risk' in objective or 'Risk' in objective:
                objectivefuncs['maximize'].append(newPathRisk)
            if 'distance' in objective or 'Distance' in objective:
                objectivefuncs['maximize'].append(newPathDistance)
            if 'fuel' in objective or 'Fuel' in objective:
                objectivefuncs['maximize'].append(newPathFuel)
            if 'catch' in objective or 'Catch' in objective:
                objectivefuncs['maximize'].append(newPathCatch)
        try:
            for const in list(constraints.keys()):
                if 'risk' in const or 'Risk' in const:
                    constraint_funcs.append(newPathRisk)
                if 'distance' in const or 'Distance' in const:
                    constraint_funcs.append(newPathDistance)
                if 'fuel' in const or 'Fuel' in const:
                    constraint_funcs.append(newPathFuel)
                if 'catch' in const or 'Catch' in const:
                    constraint_funcs.append(newPathCatch)
        except KeyError:
            constraint_funcs = []
            constraints = {}


        #objectivefuncs = {'minimize': [newPathRisk, newPathFuel, newPathDistance], 'maximize': []}#newPathCatch

        totalroutes = []
        for route in routes:
            totalroutes += [route]*len(tourments['%s' % route])
            #print(totalroutes)
        all_speed_duration = []
        for route in list(tourments.keys()):
            all_speed_duration += tourments[route]
            #print(all_speed_duration)
        #print([[all_speed_duration[i], totalroutes[i]] for i in range(len(totalroutes))])
        pareto = NSGA.pareto_optima([[all_speed_duration[i], totalroutes[i]] for i in range(len(totalroutes))], objectivefuncs=objectivefuncs)
        F = pareto[0]
        #print(F)
        sorted_solutions = []
        for f in F:
            length = len(sorted_solutions)
            length_f = len(f)
            if length < num:
                m = num - length
                n = m if m <= length_f else length_f
                distance = NSGA.crowding_distance(f, objectivefuncs)[1]
                distance.sort_values(by=['distance'], ascending=False)
                sorted_f = distance['path'].tolist()
                sorted_solutions += (sorted_f[0:n])
            else:
                break
        return sorted_solutions


    def optimal_ship_speed_duration_all_routes_2(self, ship, Graph, start, end, gene_space, objectives, constraints, generation_limits=50, num=10):
        """

        :param ship:
        :param Graph:
        :param start:
        :param end:
        :param gene_space:
        :param objectives:
        :param constraints:
        :param generation_limits:
        :param num:
        :return:
        """
        routes_graph = Graph.routes_graph(start, end)
        newgraph = routes_graph[0]
        routes = routes_graph[1]
        optimals = pandas.DataFrame()
        timedict = dict()
        nipy = plt.get_cmap('nipy_spectral')
        for i in range(0, len(routes)):
            route = routes[i]
            initial_solutions = NSGA.generate_initial_solutions(gene_space=gene_space, route=route, num=5)
            #print('initial solutions:', initial_solutions)
            newPathRisk = lambda solution: pathRisk(G=Graph.ship_timing_speed_specific_route(ship=ship,
                                                                                              speedLst=solution[0],
                                                                                              durationLst=solution[1],
                                                                                              routes_graph=newgraph,
                                                                                              route=route),
                                                    path=route)
            newPathDistance = lambda solution: pathDistance(G=Graph.ship_timing_speed_specific_route(ship=ship,
                                                                                              speedLst=solution[0],
                                                                                              durationLst=solution[1],
                                                                                              routes_graph=newgraph,
                                                                                              route=route),
                                                    path=route)

            newPathFuel = lambda solution: pathFuel(G=Graph.ship_timing_speed_specific_route(ship=ship,
                                                                                              speedLst=solution[0],
                                                                                              durationLst=solution[1],
                                                                                              routes_graph=newgraph,
                                                                                              route=route),
                                                    path=route)

            newPathCatch = lambda solution: pathCatch(G=Graph.ship_timing_speed_specific_route(ship=ship,
                                                                                              speedLst=solution[0],
                                                                                              durationLst=solution[1],
                                                                                              routes_graph=newgraph,
                                                                                              route=route),
                                                      ship=ship,
                                                      path=route)
            objectivefuncs = {'minimize': [], 'maximize': []}
            constraintfuncs = []
            for objective in objectives['minimize']:
                if 'risk' in objective or 'Risk' in objective:
                    objectivefuncs['minimize'].append(newPathRisk)
                if 'distance' in objective or 'Distance' in objective:
                    objectivefuncs['minimize'].append(newPathDistance)
                if 'fuel' in objective or 'Fuel' in objective:
                    objectivefuncs['minimize'].append(newPathFuel)
                if 'catch' in objective or 'Catch' in objective:
                    objectivefuncs['minimize'].append(newPathCatch)
            for objective in objectives['maximize']:
                if 'risk' in objective or 'Risk' in objective:
                    objectivefuncs['maximize'].append(newPathRisk)
                if 'distance' in objective or 'Distance' in objective:
                    objectivefuncs['maximize'].append(newPathDistance)
                if 'fuel' in objective or 'Fuel' in objective:
                    objectivefuncs['maximize'].append(newPathFuel)
                if 'catch' in objective or 'Catch' in objective:
                    objectivefuncs['maximize'].append(newPathCatch)
            try:
                for const in list(constraints.keys()):
                    if 'risk' in const or 'Risk' in const:
                        constraintfuncs.append(newPathRisk)
                    if 'distance' in const or 'Distance' in const:
                        constraintfuncs.append(newPathDistance)
                    if 'fuel' in const or 'Fuel' in const:
                        constraintfuncs.append(newPathFuel)
                    if 'catch' in const or 'Catch' in const:
                        constraintfuncs.append(newPathCatch)
            except KeyError:
                constraintfuncs = []
                constraints = {}

            #print('objective functions', objectivefuncs)
            #print('constraints functions', constraintfuncs)

            #newPathCatch
            starttime = time.time()
            optimal_solutions = NSGA.run_evolution_2(initial_solutions=initial_solutions, gene_space=gene_space,
                                                     objective_funcs=objectivefuncs, constraint_funcs=constraintfuncs, constraints=constraints,
                                                     generation_limit=generation_limits, num=num)
            endtime = time.time()
            #print('route %s' % route)
            #print('optimal speed and duration %s' % optimal_solutions)
            print('route', route)
            print('time used for finding optimal speed and duration:', endtime-starttime)
            print('optimal solutions for each route', optimal_solutions)
            if optimal_solutions.empty is False:
                optimal_solutions['route'] = [route]*len(optimal_solutions)
            optimals = pandas.concat([optimals, optimal_solutions])
        #     timedict['%s' % route] = endtime-starttime
        #     risk = [objectivefuncs['minimize'][0](optimal_solutions[i]) for i in range(0, len(optimal_solutions))]
        #     fuel = [objectivefuncs['maximize'][0](optimal_solutions[i]) for i in range(0, len(optimal_solutions))]
        #     plt.scatter(risk, fuel, color=nipy(i / 100), marker="s", label='Route: %s' % i)
        # plt.legend(fontsize=5)
        # plt.show()
        # plt.plot(timedict.keys(), timedict.values())
        # plt.show()
        return optimals

    def get_optimal_route_from_all_routes_2(self, ship, Graph, start, end, gene_space, objectives, constraints, resultdir, generation_limits=50, num=10):
        routes_graph = Graph.routes_graph(start, end)
        newgraph = routes_graph[0]
        routes = routes_graph[1]
        tourments = self.optimal_ship_speed_duration_all_routes_2(ship, Graph, start, end, gene_space, objectives, constraints, generation_limits=generation_limits, num=num)
        #print(tourments)
        tourments = tourments.reset_index(drop=True)
        tourments.rename(columns={'path':'speed_duration'}, inplace=True)
        tourments['path'] = range(0, len(tourments))
        minimizationObj = ['minimizationObj %s' % i for i in range(0, len(objectives['minimize']))]
        maximizationObj = ['maximizationObj %s' % i for i in range(0, len(objectives['maximize']))]
        objectives = {'minimize': minimizationObj, 'maximize': maximizationObj}
        tourments = NSGA.constraint_handling_2(tourments, constraints)
        print('tourments', tourments)
        pareto = NSGA.pareto_optima_2(tourments, objectives=objectives)
        solutionsinfo = pareto[1]
        solutionsinfo.to_csv('%s/paths %s.csv' % (resultdir, routes[0][0]))
        print('final', solutionsinfo)
        F = pareto[0]
        # print('pareto front:', F)
        sorted_solutionsinfo = pandas.DataFrame()
        num = len(solutionsinfo[solutionsinfo['pathfront'] == 1])
        for k in range(1, len(F)+1):
            f = solutionsinfo[solutionsinfo['pathfront'] == k]
            length = len(sorted_solutionsinfo)
            length_f = len(f)
            if length <= num:
                m = num - length
                n = m if m <= length_f else length_f
                sorted_f = NSGA.crowding_distance_2(f, objectives)
                sorted_solutionsinfo= pandas.concat([sorted_solutionsinfo, sorted_f.head(n)])
            else:
                break
        if sorted_solutionsinfo.empty is False:
            all_speed_duration = sorted_solutionsinfo['speed_duration'].tolist()
            totalroutes = sorted_solutionsinfo['route'].tolist()
            optimals = [[all_speed_duration[i], totalroutes[i]] for i in range(len(totalroutes))]
        else:
            print('No route is within the constraints.')
            exit()
        sorted_solutionsinfo.to_csv('%s/sorted_paths %s.csv' % (resultdir, routes[0][0]))
        return optimals


class Gear:
    """Decision maker to decide how much fish to get from a gear drop."""


class Visulization:
    """For visualization"""
    def __init__(self, map, fishlocation, fishfactory, home, obstacles):
        self.map=map
        self.fishlocation = fishlocation
        self.fishfactory = fishfactory
        self.home = home
        self.obstacles = obstacles

    def visulize(self):
        return



def retrieve_point_weather(Weather_map, position, time):
    locweather = Weather_map(position, time)
    return locweather


def retrieve_passage_weather(Weather_map, startLoc, endLoc, startTime, endTime):
    locweather = Weather_map(startLoc, endLoc, startTime, endTime)
    return locweather


def retrieve_point_duration_weather(Weather_map, position, startTime, endTime):
    locweather = Weather_map(position, startTime, endTime)
    return locweather


def passageRisk(Weather_map, startLoc, endLoc, startTime, endTime, hazards, ship):
    locweather= retrieve_passage_weather(Weather_map,Weather_map, startLoc, endLoc, startTime, endTime)
    risk = ship *locweather
    return risk

def passageRisk_trial(ship, speed, distance):
    risk = 0.5 * distance/speed
    return risk

def locRisk(Weather_map, position, startTime, endTime, hazards, ship):
    locweather = retrieve_point_duration_weather(Weather_map, position, startTime, endTime)
    risk = ship*hazards*locweather
    return risk

def locRisk_trial(ship, duration):
    risk = duration*100
    return risk

def passageFuel_Consumption(ship, distance, speed):
    fuelCon = distance*speed*0.01
    return fuelCon

def locFuel_Consumption(ship, duration):
    fuelCon = duration*15
    return fuelCon

def great_circle_distance(startLoc:tuple, endLoc:tuple):
    lat1 = startLoc[0]
    lon1 = startLoc[1]
    lat2 = endLoc[0]
    lon2 = endLoc[1]
    #distance = math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(abs(lon2 - lon1))) * 6371 #(6371 is Earth radius in km.)
    distance = geopy.distance.great_circle(startLoc, endLoc).km
    return distance

def searoute_distance(startLoc:tuple, endLoc:tuple):
    lat1 = startLoc[0]
    lon1 = startLoc[1]
    lat2 = endLoc[0]
    lon2 = endLoc[1]
    startLoc, endLoc=[startLoc[0], startLoc[1]], [endLoc[0], endLoc[1]]
    distance = searoute.searoute(origin=startLoc, destination=endLoc)['properties']['length']
    return distance

def scgraph_distance(startLoc:tuple, endLoc:tuple):
    distance = marnet_geograph.get_shortest_path(origin_node={"latitude": startLoc[0], "longitude": startLoc[1]},
                                                 destination_node={"latitude": endLoc[0], "longitude": endLoc[1]}
                                                 )
    # Show your output path
    print(str([[i['latitude'], i['longitude']] for i in distance['coordinate_path']]))

    # Show the length
    print('Length: ', distance['length'])  # => Length:  19596.4653
    return distance['length']


def pathAttribute(G, path, attribute, method=None):
    """
    To calculate the total risk of a path from all nodes and edges in the path.
    :param G: graph
    :param path: paths
    :param method: specify whether to calculate mean, median, or a percentile value from the probabilistic risk distribution if it is a distribution.
    Method can be "Median" or "median", "mean" or "Median", or any percentile number such as 10, 90
    :return: total value of the specified attribute of the path
    """
    attribute = attribute
    G = G.copy()
    j = 0
    if method == 'Median' or method == 'median':
        while j <= len(path)-1:
            try:
                node_attr = G.nodes[path[j]][attribute]
                if (isinstance(node_attr, int) or isinstance(node_attr, float)) is False:
                    node_attr = node_attr.median()
            except KeyError:
                nx.set_node_attributes(G, {path[j]: {attribute: 0}})
                if j < len(path)-1:
                    try:
                        edge_attr = G.edges[path[j], path[j+1]][attribute]
                        if (isinstance(edge_attr, int) or isinstance(edge_attr, float)) is False:
                            edge_attr = edge_attr.median()
                    except KeyError:
                        nx.set_edge_attributes(G, {(path[j], path[j+1]): {attribute: 0}})
            j += 1
    elif method == 'Mean' or method == 'mean':
        while j <= len(path)-1:
            try:
                node_attr = G.nodes[path[j]][attribute]
                if (isinstance(node_attr, int) or isinstance(node_attr, float)) is False:
                    node_attr = node_attr.mean()
            except KeyError:
                nx.set_node_attributes(G, {path[j]: {attribute: 0}})
                if j < len(path)-1:
                    try:
                        edge_attr = G.edges[path[j], path[j+1]][attribute]
                        if (isinstance(edge_attr, int) or isinstance(edge_attr, float)) is False:
                            edge_attr = edge_attr.mean()
                    except KeyError:
                        nx.set_edge_attributes(G, {(path[j], path[j+1]): {attribute: 0}})
            j += 1
    elif isinstance(method, int) or isinstance(method, float):
        while j <= len(path) - 1:
            try:
                node_attr = G.nodes[path[j]][attribute]
                if (isinstance(node_attr, int) or isinstance(node_attr, float)) is False:
                    node_attr = node_attr.ppf(method)
            except KeyError:
                nx.set_node_attributes(G, {path[j]: {attribute: 0}})
                if j < len(path) - 1:
                    try:
                        edge_attr = G.edges[path[j], path[j + 1]][attribute]
                        if (isinstance(edge_attr, int) or isinstance(edge_attr, float)) is False:
                            edge_attr = edge_attr.ppf(method)
                    except KeyError:
                        nx.set_edge_attributes(G, {(path[j], path[j + 1]): {attribute: 0}})
            j += 1
    else:
        while j <= len(path)-1:
            try:
                node_attr = G.nodes[path[j]][attribute]
                if (isinstance(node_attr, int) or isinstance(node_attr, float)) is False:
                    node_attr = node_attr.mean()
            except KeyError:
                nx.set_node_attributes(G, {path[j]: {attribute: 0}})
                if j < len(path)-1:
                    try:
                        edge_attr = G.edges[path[j], path[j+1]][attribute]
                        if (isinstance(edge_attr, int) or isinstance(edge_attr, float)) is False:
                            edge_attr = edge_attr.mean()
                    except KeyError:
                        nx.set_edge_attributes(G, {(path[j], path[j+1]): {attribute: 0}})
            j += 1
    totalvalue = nx.path_weight(G, path, weight=attribute) + sum([G.nodes[path[j]][attribute] for j in range(0, len(path))])
    return totalvalue


def pathAttribute_nodes(G, path, attribute, method=None):
    """
    To calculate the total risk of a path from all nodes in the path.
    :param G: graph
    :param path: paths
    :param method: specify whether to calculate mean, median, or a percentile value from the probabilistic risk distribution if it is a distribution.
    Method can be "Median" or "median", "mean" or "Median", or any percentile number such as 10, 90
    :return: total value of the specified attribute of the path
    """
    attribute = attribute
    G = G.copy()
    j = 0
    if method == 'Median' or method == 'median':
        while j <= len(path)-2:
            try:
                node_attr = G.nodes[path[j]][attribute]
                if (isinstance(node_attr, int) or isinstance(node_attr, float)) is False:
                    node_attr = node_attr.median()
            except KeyError:
                nx.set_node_attributes(G, {path[j]: {attribute: 0}})
            j += 1
    elif method == 'Mean' or method == 'mean':
        while j <= len(path)-2:
            try:
                node_attr = G.nodes[path[j]][attribute]
                if (isinstance(node_attr, int) or isinstance(node_attr, float)) is False:
                    node_attr = node_attr.mean()
            except KeyError:
                nx.set_node_attributes(G, {path[j]: {attribute: 0}})
            j += 1
    elif isinstance(method, int) or isinstance(method, float):
        while j <= len(path) - 2:
            try:
                node_attr = G.nodes[path[j]][attribute]
                if (isinstance(node_attr, int) or isinstance(node_attr, float)) is False:
                    node_attr = node_attr.ppf(method)
            except KeyError:
                nx.set_node_attributes(G, {path[j]: {attribute: 0}})
            j += 1
    else:
        while j <= len(path)-2:
            try:
                node_attr = G.nodes[path[j]][attribute]
                if (isinstance(node_attr, int) or isinstance(node_attr, float)) is False:
                    node_attr = node_attr.mean()
            except KeyError:
                nx.set_node_attributes(G, {path[j]: {attribute: 0}})
            j += 1
    totalvalue = sum([G.nodes[path[j]][attribute] for j in range(0, len(path))])
    return totalvalue

def pathAttribute_edges(G, path, attribute, method=None):
    """
    To calculate the total risk of a path from all edges in the path.
    :param G: graph
    :param path: paths
    :param method: specify whether to calculate mean, median, or a percentile value from the probabilistic risk distribution if it is a distribution.
    Method can be "Median" or "median", "mean" or "Median", or any percentile number such as 10, 90
    :return: total value of the specified attribute of the path
    """

    attribute = attribute
    G = G.copy()
    j = 0
    if method == 'Median' or method == 'median':
        while j < len(path)-1:
            try:
                edge_attr = G.edges[path[j], path[j+1]][attribute]
                if (isinstance(edge_attr, int) or isinstance(edge_attr, float)) is False:
                    edge_attr = edge_attr.median()
            except KeyError:
                nx.set_edge_attributes(G, {(path[j], path[j+1]): {attribute: 0}})
            j += 1
    elif method == 'Mean' or method == 'mean':
        while j < len(path)-1:
            try:
                edge_attr = G.edges[path[j], path[j+1]][attribute]
                if (isinstance(edge_attr, int) or isinstance(edge_attr, float)) is False:
                    edge_attr = edge_attr.mean()
            except KeyError:
                nx.set_edge_attributes(G, {(path[j], path[j+1]): {attribute: 0}})
            j += 1
    elif isinstance(method, int) or isinstance(method, float):
        while j < len(path) - 1:
            try:
                edge_attr = G.edges[path[j], path[j + 1]][attribute]
                if (isinstance(edge_attr, int) or isinstance(edge_attr, float)) is False:
                    edge_attr = edge_attr.ppf(method)
            except KeyError:
                nx.set_edge_attributes(G, {(path[j], path[j + 1]): {attribute: 0}})
            j += 1
    else:
        while j <= len(path)-1:
            try:
                edge_attr = G.edges[path[j], path[j+1]][attribute]
                if (isinstance(edge_attr, int) or isinstance(edge_attr, float)) is False:
                    edge_attr = edge_attr.mean()
            except KeyError:
                nx.set_edge_attributes(G, {(path[j], path[j+1]): {attribute: 0}})
            j += 1
    totalvalue = sum([G.nodes[path[j]][attribute] for j in range(0, len(path))])
    return totalvalue


def pathDistance(G, path, method=None):
    """
    To calculate the total distance of a path from all nodes and edges in the path.
    :param G: graph
    :param path: paths
    :param method: specify whether to calculate mean, median, or a percentile value from the probabilistic risk distribution if it is a distribution.
    Method can be "Median" or "median", "mean" or "Median", or any percentile number such as 10, 90
    :return: total distance of the path
    """
    totaldistance = pathAttribute(G, path, 'distance', method)
    return totaldistance

def pathDuration(G, path, method=None):
    """
    To calculate the total distance of a path from all nodes and edges in the path.
    :param G: graph
    :param path: paths
    :param method: specify whether to calculate mean, median, or a percentile value from the probabilistic risk distribution if it is a distribution.
    Method can be "Median" or "median", "mean" or "Median", or any percentile number such as 10, 90
    :return: total distance of the path
    """
    totalduration = pathAttribute(G, path, 'duration', method)
    return totalduration


def pathCatch(ship, G, path, method=None):
    """
    To calculate the total distance of a path from all nodes and edges in the path.
    :param G: graph
    :param path: paths
    :param method: specify whether to calculate mean, median, or a percentile value from the probabilistic risk distribution if it is a distribution.
    Method can be "Median" or "median", "mean" or "Median", or any percentile number such as 10, 90
    :return: total distance of the path
    """
    fishtype = ship.targeted_fish
    targeted_quantity = ship.remaining_quantity
    G = G.copy()
    totalCatch = 0
    j = 0
    if method == 'Median' or method == 'median':
        while j <= len(path) - 2:
            try:
                node_attr = G.nodes[path[j]]['fishstock']
                maximum_catch = node_attr[fishtype]['quantity']
                if (isinstance(maximum_catch, int) or isinstance(maximum_catch, float)) is False:
                    maximum_catch = maximum_catch.median()
            except KeyError:
                maximum_catch = 0
            j += 1
            difference = targeted_quantity - totalCatch
            if difference <= maximum_catch:
                totalCatch += difference
            else:
                totalCatch = +maximum_catch
    elif method == 'Mean' or method == 'mean':
        while j <= len(path) - 2:
            try:
                node_attr = G.nodes[path[j]]['fishstock']
                maximum_catch = node_attr[fishtype]['quantity']
                if (isinstance(maximum_catch, int) or isinstance(maximum_catch, float)) is False:
                    maximum_catch = maximum_catch.mean()
            except KeyError:
                maximum_catch = 0
            j += 1
            totalCatch += maximum_catch
    elif isinstance(method, int) or isinstance(method, float):
        while j <= len(path) - 2:
            try:
                node_attr = G.nodes[path[j]]['fishstock']
                maximum_catch = node_attr[fishtype]['quantity']
                if (isinstance(maximum_catch, int) or isinstance(maximum_catch, float)) is False:
                    maximum_catch = maximum_catch.ppf(method)
            except KeyError:
                maximum_catch = 0
            j += 1
            difference = targeted_quantity - totalCatch
            if difference <= maximum_catch:
                totalCatch += difference
            else:
                totalCatch += maximum_catch
    else:
        while j <= len(path) - 1:
            try:
                node_attr = G.nodes[path[j]]['fishstock']
                maximum_catch = node_attr[fishtype]['quantity']
                if (isinstance(maximum_catch, int) or isinstance(maximum_catch, float)) is False:
                    maximum_catch = maximum_catch.mean()
            except KeyError:
                maximum_catch = 0
            j += 1
            difference = targeted_quantity - totalCatch
            if difference <= maximum_catch:
                totalCatch += difference
            else:
                totalCatch += maximum_catch
    return totalCatch


def pathRisk(G, path, method=None):
    """
    To calculate the total risk of a path from all nodes and edges in the path.
    :param G: graph
    :param path: paths
    :param method: specify whether to calculate mean, median, or a percentile value from the probabilistic risk distribution if it is a distribution.
    Method can be "Median" or "median", "mean" or "Median", or any percentile number such as 10, 90
    :return: total risk of the path
    """
    totalrisk = pathAttribute(G, path, 'risk', method)
    #print('total risk', totalrisk)
    return totalrisk


def pathGain(self, graph, path):
    G = graph
    pathgain = sum([4 * (G.nodes[path[j]]['fish stock'] + G.nodes[path[j + 1]]['fish stock']) / 2 - 0.5 *
         G.edges[path[j], path[j + 1]]['risk'] - 0.01 * G.edges[path[j], path[j + 1]][
             'distance'] for j in range(0, len(path) - 1)])
    return pathgain


def pathCost(self, graph, path):
    G = graph
    pathCost = sum([4 * (G.nodes[path[j]]['fish stock'] + G.nodes[path[j + 1]]['fish stock']) / 2 - 0.5 *
         G.edges[path[j], path[j + 1]]['risk'] - 0.01 * G.edges[path[j], path[j + 1]][
             'distance'] for j in range(0, len(path) - 1)])
    return pathCost





def pathFuel(G, path, method=None):
    totalfuel = pathAttribute(G, path, 'fuel consumption', method)
    #print('total fuel', totalfuel)
    return totalfuel



def next_loc(self, optimal_routes):
    if len(optimal_routes) == 1:
        loc = optimal_routes[1]
    else:
        loc_candidates = list(set([optimal_routes[i][1] for i in range(0, len(optimal_routes))]))
        all_locs = [optimal_routes[i][1] for i in range(0, len(optimal_routes))]




def generate_waypoint(self, start, end, resolution, map):
    return []


def dynamic_routing(Graph, ship, gene_space, start, end, objectives, constraints):
    optimizor = RouteOptimizer()
    ship = ship
    graph = Graph
    current_place = start
    Locations = []
    Speeds = []
    Durations = []
    Edges = []
    Footprint = nx.DiGraph()
    catch = 0
    while current_place != end:
        "Update the total catch of the ship and fish stock of the fishground."
        if isinstance(current_place, FishGround):
            fishtype = ship.targeted_fish
            catch = ship.update_catch(current_place)
            stock = current_place.fishstock
            stock[fishtype]['quantity'] -= catch
            current_place.update_stock(stock)
            graph.update_node_attribute(current_place, 'fishstock', stock)
            print('catch from current fishground:', catch)
            print('total accumulated catch of current ship:', ship.catch)
            print('remaining fish stock:', current_place.fishstock)
        "Here we need to update objectives"
        try:
            constraints['total catch']['low'] = max(constraints['total catch']['low'] - catch, 0)
            constraints['total catch']['high'] = max(constraints['total catch']['high'] - catch, 0)
            print('new constraints', constraints)
        except KeyError:
            pass
        start = current_place
        print('current place', current_place.name)
        Locations.append(start)
        end = end
        optimal = optimizor.get_optimal_route_from_all_routes_2(ship=ship,
                                                              Graph=graph,
                                                              start=start,
                                                              end=end,
                                                              gene_space=gene_space,
                                                              objectives=objectives,
                                                              constraints=constraints,
                                                              generation_limits=5,
                                                              num=5)

        print('all optimal routes', optimal)
        routes = [route[1] for route in optimal]
        candidates = [stop[1] for stop in routes]
        routes_graph = graph.routes_graph(start, end)[0]
        print(routes_graph)
        candidate_objects = [routes_graph.nodes[place]['object'].name for place in candidates]
        print('candidate object', candidate_objects)
        next_place = mode(candidate_objects)
        print('locations:', Locations)
        location_filtered = list(filter(lambda route: next_place in route[1][1], optimal))
        print('filtered_routes', location_filtered)
        print('speed candidiates', [solution[0][0][0] for solution in location_filtered])
        speed = mode([solution[0][0][0] for solution in location_filtered])
        print('speed', speed)
        speed_filtered = list(filter(lambda solution: solution[0][0][0] == speed, location_filtered))
        Speeds.append(speed)
        duration = 0 if next_place == 'E' else mode([solution[0][1][0] for solution in speed_filtered])
        print('duration', duration)
        Durations.append(duration)
        current_place = list(filter(lambda n: n.name == next_place, list(graph.graph.nodes())))[0]
        Edges.append((start, current_place))
        if next_place == 'E':
            Locations.append(end)
            break
    return Locations, Speeds, Durations

def Animate_dynamic_route(Graph, Locations, Speeds, Durations):
    Footprint = nx.DiGraph()
    graph = Graph
    fig = plt.figure()
    graph.plot_graph()
    def initial_graph():
        current_place = Locations[0]
        nodeAttr = graph.graph.nodes[current_place].copy()
        Footprint.add_node(current_place, attr=nodeAttr)
        pos = {n: [n.position[1], n.position[0]] for n in list(Footprint.nodes())}
        nx.draw(Footprint, pos=pos, node_color='red')

    def animate(index):
        if index == 0:
            current_place = Locations[0]
            nodeAttr = graph.graph.nodes[current_place].copy()
            Footprint.add_node(current_place, attr=nodeAttr)
            pos = {n: [n.position[1], n.position[0]] for n in list(Footprint.nodes())}
            nx.draw(Footprint, pos=pos, node_color='red')
        else:
            index = index-1
            current_place = Locations[index]
            next_place = Locations[index + 1]
            nodeAttr = graph.graph.nodes[next_place].copy()
            nodeAttr['Duration'] = Durations[index]
            Footprint.add_node(next_place)
            nx.set_node_attributes(Footprint, {next_place: nodeAttr})
            edgeAttr = graph.graph.edges[(current_place, next_place)].copy()
            edgeAttr['Speed'] = Speeds[index]
            Footprint.add_edge(current_place, next_place)
            nx.set_edge_attributes(Footprint, {(current_place, next_place): edgeAttr})
            width = [Footprint[u][v]['Speed'] if Footprint[u][v]['Speed'] <= 10 else 10 for u, v in Footprint.edges()]
            pos = {n: [n.position[1], n.position[0]] for n in list(Footprint.nodes())}
            #labels = {n: graph.graph.nodes[n]['fishstock']['simon']['quantity'] for n in list(Footprint.nodes())}
            nx.draw(Footprint, pos=pos, node_color='red', edge_color='red', width=width)
            pos_for_speed = {n: [n.position[1]-5, n.position[0]-5] for n in list(Footprint.nodes())}
            nx.draw_networkx_edge_labels(Footprint, pos_for_speed,
                                         edge_labels={(u, v): "Speed:{:.0f}".format(d['Speed']) for u, v, d in
                                                      Footprint.edges(data=True)}, font_color='red')
            nodes = []
            for node in list(Footprint.nodes()):
                state = isinstance(node, Start) or isinstance(node, End)
                if state is False:
                    nodes.append(node)
            nx.draw_networkx_labels(Footprint, pos_for_speed,
                                    labels={n: "Duration:{:.0f}".format(Footprint.nodes[n]['Duration']) for n in nodes},
                                    font_color='red',
                                    font_size=9,
                                    horizontalalignment='center')

        #nx.draw_networkx_labels(Footprint, pos=pos, labels=labels)
        #nx.draw_networkx_edges(Footprint, pos=pos, edge_color='red', width=width)


    ani = animation.FuncAnimation(fig, animate, frames=range(0, len(Locations)), interval=1000, init_func=initial_graph)
    ani.save(f'route {Locations[0].name}-{Locations[-1].name}.gif', writer=PillowWriter(fps=1))


def dynamic_routing_with_visualization(Graph, ship, gene_space, start, end, objectives, constraints,resultdir):
    graph =Graph
    ship=ship
    gene_space=gene_space
    Footprint=nx.DiGraph()
    optimizor = RouteOptimizer()
    fig = plt.figure()
    Locations = [start]
    Speeds = []
    Durations = []
    Edges = []
    def initial_draw():
        current_place = Locations[0]
        graph.plot_graph()

        nodeAttr = graph.graph.nodes[current_place].copy()
        Footprint.add_node(current_place, attr=nodeAttr)
        pos = {n: [n.position[1], n.position[0]] for n in list(Footprint.nodes())}
        nx.draw(Footprint, pos=pos, node_color='red')

    def animate(index, end=end):
        if index == 0:
            current_place = Locations[0]
            nodeAttr = graph.graph.nodes[current_place].copy()
            Footprint.add_node(current_place, attr=nodeAttr)
            pos = {n: [n.position[1], n.position[0]] for n in list(Footprint.nodes())}
            nx.draw(Footprint, pos=pos, node_color='red')
        else:
            index = index-1
            current_place = Locations[index]
            if current_place == end:
                exit()
            catch = 0
            "Update the total catch of the ship and fish stock of the fishground."
            if isinstance(current_place, FishGround):
                fishtype = ship.targeted_fish
                catch = ship.update_catch(current_place)
                stock = current_place.fishstock
                stock[fishtype]['quantity'] -= catch
                current_place.update_stock(stock)
                graph.update_node_attribute(current_place, 'fishstock', stock)
                print('catch from current fishground:', catch)
                print('total accumulated catch of current ship:', ship.catch)
                print('remaining fish stock:', current_place.fishstock)
            "Here we need to update objectives"
            try:
                constraints['total catch']['low'] = max(constraints['total catch']['low'] - catch, 0)
                constraints['total catch']['high'] = max(constraints['total catch']['high'] - catch, 0)
                print('new constraints', constraints)
            except KeyError:
                pass
            #print('current place', current_place.name)
            end = end
            optimal = optimizor.get_optimal_route_from_all_routes_2(ship=ship,
                                                                  Graph=graph,
                                                                  start=current_place,
                                                                  end=end,
                                                                  gene_space=gene_space,
                                                                  objectives=objectives,
                                                                  constraints=constraints,
                                                                  generation_limits=150,
                                                                  num=5, resultdir=resultdir)
            print('all routes', optimal)
            routes = [route[1] for route in optimal]
            print('all routes:', routes)
            candidates = [stop[1] for stop in routes]
            print('candidates', candidates)
            routes_graph = graph.routes_graph(current_place, end)[0]
            print(routes_graph)
            candidate_objects = [routes_graph.nodes[place]['object'].name for place in candidates]
            print('candidate object', candidate_objects)
            next_place = mode(candidate_objects)
            next_location = list(filter(lambda n: n.name == next_place, list(graph.graph.nodes())))[0]
            Locations.append(next_location)
            print('locations:', Locations)
            location_filtered = list(filter(lambda route: next_place in route[1][1], optimal))
            print('filtered_routes', location_filtered)
            speed = mode([solution[0][0][0] for solution in location_filtered])
            speed_filtered = list(filter(lambda solution: solution[0][0][0] == speed, location_filtered))
            print('speed_filtered', speed_filtered)
            print('speed', speed)
            Speeds.append(speed)
            duration = 0 if next_place == 'E' else mode([solution[0][1][0] for solution in speed_filtered])
            Durations.append(duration)
            print('duration', duration)
            Edges.append((current_place, next_location))
            print('edge:', current_place.name, next_location.name)
            nodeAttr = graph.graph.nodes[next_location].copy()
            nodeAttr['Duration'] = duration
            Footprint.add_node(next_location)
            nx.set_node_attributes(Footprint, {next_location: nodeAttr})
            edgeAttr = graph.graph.edges[(current_place, next_location)].copy()
            edgeAttr['Speed'] = speed
            Footprint.add_edge(current_place, next_location)
            nx.set_edge_attributes(Footprint, {(current_place, next_location): edgeAttr})
            plt.cla()
            graph.plot_graph()
            pos = {n: [n.position[1], n.position[0]] for n in list(Footprint.nodes())}
            graph.plot_optimals(ship=ship, optimized_routes=optimal, routes_graph=routes_graph, num=len(optimal))
            nx.draw_networkx_nodes(Footprint, pos=pos, nodelist=[current_place], node_color='red')
            plt.savefig(fname='%s/%s_multi_routes.png' % (resultdir, next_place))
            width = [Footprint[u][v]['Speed'] if Footprint[u][v]['Speed'] <= 5 else 5 for u, v in Footprint.edges()]
            pos = {n: [n.position[1], n.position[0]] for n in list(Footprint.nodes())}
            nx.draw_networkx(Footprint, with_labels=False, pos=pos, node_color='red', edge_color='red', width=width, font_color='red')
            nx.draw_networkx_edge_labels(Footprint, pos,
                                         edge_labels={(u, v): "D:{:.0f}\nS:{:.0f}".format(d['distance'], d['Speed']) for u, v, d in
                                                      Footprint.edges(data=True)}, font_color='red')
            nodes = []
            pos_for_duration = {n: [n.position[1]+6, n.position[0]] for n in list(Footprint.nodes())}
            for node in list(Footprint.nodes()):
                state = isinstance(node, Start) or isinstance(node, End)
                if state is False:
                    nodes.append(node)
            nx.draw_networkx_labels(Footprint, pos_for_duration,
                                    labels={n: "Du:{:.0f}".format(Footprint.nodes[n]['Duration']) for n in nodes},
                                    font_color='red',
                                    font_size=9,
                                    horizontalalignment='center')
            plt.savefig(fname='%s/%s_next_action.png' % (resultdir, next_place))
    ani = animation.FuncAnimation(fig, animate, init_func=initial_draw, interval=2000)
    #plt.show()
    ani.save(f'{resultdir}/route {start.name}-{end.name}.gif', writer=PillowWriter(fps=1))




