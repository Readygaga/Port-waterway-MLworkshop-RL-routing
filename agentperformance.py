import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
from main import Graph, FishGround, FishFactory, WayPoint, Start, End, Position, great_circle_distance, \
    scgraph_distance, pathRisk, pathCatch, pathGain, pathFuel, pathDistance, Ship, dynamic_routing
import networkx as nx

plt.rcParams['font.family'] = 'Times New Roman'
log_dir = 'Training/Logs/obj1_2millionstepModel/PPO_1'
log_dir = 'Training/Logs/masked_obj2_2millionstepModel_900/PPO_1'

event_acc = event_accumulator.EventAccumulator(log_dir)
event_acc.Reload()

# List all tags
tags = event_acc.Tags()['scalars']
print("Available tags:")
print(tags)
available_tags = ['rollout/ep_len_mean', 'rollout/ep_rew_mean', 'time/fps', 'train/approx_kl', 'train/clip_fraction',
                  'train/clip_range', 'train/entropy_loss', 'train/explained_variance', 'train/learning_rate',
                  'train/loss', 'train/policy_gradient_loss', 'train/value_loss']

logs = ['Training/Logs/obj1_2millionstepModel/PPO_1', 'Training/Logs/masked_obj2_2millionstepModel_900/PPO_1',
        'Training/Logs/masked_obj3_3millionstepModel2/PPO_2',
        'Training/Logs/masked_obj4_10millionstepModel_revisedrisk/PPO_2',
        'Training/Logs/masked_obj5_5millionstepModel/PPO_4']
labels = ['(a) Obj 1', '(b) Obj 2', '(c) Obj 3', '(d) Obj 4', '(e) Obj 5']
ylabels = ['Negative of \n accumulated distance', 'Accumulated catch', 'Negative of \n accumulated distance',
           'Negative of risk', 'Accumulated Catch-Risk']
xticks = [['0', '0.5M', '1M', '1.5M', '2M'],
          ['0', '0.5M', '1M', '1.5M', '2M', '2.5M', '3M'],
          ['0', '0.5M', '1M', '1.5M', '2M', '2.5M', '3M'],
          ['0', '2M', '4M', '6M', '8M', '10M'],
          ['0', '1M', '2M', '3M', '4M', '5M']]


# Function to extract data from TensorBoard log
def extract_data(log_dir):
    data = []
    for log in log_dir:
        event_acc = event_accumulator.EventAccumulator(log)
        event_acc.Reload()
        # Retrieve data
        steps = []
        rewards = []
        for tag in event_acc.Tags()['scalars']:
            if tag.endswith('/ep_rew_mean'):  # Adjust this based on your tag names
                steps_and_rewards = event_acc.Scalars(tag)
                steps = [scalar.step for scalar in steps_and_rewards]
                rewards = [scalar.value for scalar in steps_and_rewards]
        data.append((steps, rewards))

    return data


# Function to plot data
def plot_data(data):
    plt.figure(figsize=(5, 12))
    plt.suptitle('Accumulated reward per episode', x=0.6)
    N = len(data)
    j = 0
    for D in data:
        plt.subplot(N, 1, j + 1)
        steps = D[0]
        rewards = D[1]
        plt.plot(steps, rewards, label=labels[j], color='blue', linewidth=0.5)
        plt.ylabel(ylabels[j])
        plt.xticks(ticks=np.arange(0, steps[-1] + 1, step=steps[-1] / (len(xticks[j]) - 1)), labels=xticks[j])
        # plt.text(x=steps[-1], y=min(rewards), s=labels[j], horizontalalignment='right')
        # plt.title('Reward over Steps')
        j += 1
    # plt.legend()
    # plt.suptitle('Reward over Steps')
    plt.xlabel('Steps')
    plt.tight_layout()
    plt.savefig(fname='performance.png')
    plt.show()


# Extract and plot the data
# data = extract_data(logs)
# plot_data(data)

def plot_data2(data):
    N = len(data)
    j = 0
    for D in data:
        plt.figure(figsize=(5, 2.5))
        steps = D[0]
        rewards = D[1]
        plt.plot(steps, rewards, label='Accumulated Reward per Episode', color='blue', linewidth=0.5)
        plt.ylabel(ylabels[j])
        plt.xticks(ticks=np.arange(0, steps[-1] + 1, step=steps[-1] / (len(xticks[j]) - 1)), labels=xticks[j])
        plt.text(x=steps[-1], y=min(rewards), s=labels[j], horizontalalignment='right')
        # plt.title('Reward over Steps')
        plt.tight_layout()
        plt.savefig(fname='%s.png' % labels[j])
        j += 1
    # plt.legend()
    # plt.suptitle('Reward over Steps')
    # plt.xlabel('Steps')
    plt.show()


# Extract and plot the data
#data = extract_data(logs)
#plot_data2(data)


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


def draw_fishing_network():
    pos = {n: [n.position[1], n.position[0]] for n in list(graph1.graph.nodes)}
    nx.draw(graph1.graph, pos, node_size=300)
    nx.draw_networkx_labels(graph1.graph, pos, labels={n: n.name for n in list(graph1.graph.nodes)}, font_size=10,
                            font_family='Times New Roman')
    fishground = list(filter(lambda node: isinstance(node, FishGround), list(graph1.graph.nodes())))
    stockPos = {n: [n.position[1] + 3, n.position[0] + 3] for n in list(fishground)}
    position = [[12, 35], [55, 25], [57, 62]]
    stockPos = {n: position[j] for n, j in zip(list(fishground), [0, 1, 2])}
    nx.draw_networkx_labels(graph1.graph, stockPos,
                            labels={n: 'Fishstock:{}'.format(n.fishstock['Sei']['quantity']) for n in fishground},
                            font_color='green', font_size=9, font_family='Times New Roman')
    edges_without_labels = [(loc2, ff2), (loc3, ff2)]
    edges_with_labels = list(filter(lambda edge: edge not in edges_without_labels, graph1.graph.edges))
    nx.draw_networkx_edge_labels(graph1.graph, pos,
                                 edge_labels={d: "D:{:.0f}".format(graph1.graph.edges[d]['distance']) for d in
                                              edges_with_labels},
                                 font_size=9, font_family='Times New Roman')
    plt.savefig(fname='fishing network.png')


def draw_route1(DR):
    pos = {n: [n.position[1], n.position[0]] for n in list(graph1.graph.nodes)}
    nx.draw(graph1.graph, pos, node_size=300)
    nx.draw_networkx_labels(graph1.graph, pos, labels={n: n.name for n in list(graph1.graph.nodes)}, font_size=10,
                            font_family='Times New Roman')
    fishground = list(filter(lambda node: isinstance(node, FishGround), list(graph1.graph.nodes())))
    stockPos = {n: [n.position[1] + 3, n.position[0] + 3] for n in list(fishground)}
    position = [[12, 35], [55, 25], [57, 62]]
    stockPos = {n: position[j] for n, j in zip(list(fishground), [0, 1, 2])}
    nx.draw_networkx_labels(graph1.graph, stockPos,
                            labels={n: 'Fishstock:{}'.format(n.fishstock['Sei']['quantity']) for n in fishground},
                            font_color='green', font_size=9, font_family='Times New Roman')
    edges_without_labels = [(loc2, ff2), (loc3, ff2)]
    edges_with_labels = list(filter(lambda edge: edge not in edges_without_labels, graph1.graph.edges))
    nx.draw_networkx_edge_labels(graph1.graph, pos,
                                 edge_labels={d: "D:{:.0f}".format(graph1.graph.edges[d]['distance']) for d in
                                              edges_with_labels},
                                 font_size=9, font_family='Times New Roman')
    pos = {n: [n.position[1], n.position[0]] for n in list(graph1.graph.nodes())}
    nx.draw_networkx_nodes(G=graph1.graph, pos=pos, nodelist=DR[0], node_color='red', node_size=300)
    edgelist1 = [(DR[0][i], DR[0][i + 1]) for i in range(len(DR[0]) - 1)]
    edgelist = list(filter(lambda edge: edge not in edges_without_labels, edgelist1))
    nx.draw_networkx_edges(G=graph1.graph, pos=pos, edgelist=edgelist, edge_color='red')
    distance = [graph1.graph.edges[edge]['distance'] for edge in edgelist]
    nx.draw_networkx_edge_labels(G=graph1.graph,
                                 pos=pos,
                                 edge_labels={edge: "D:{:.0f}".format(distance) for
                                              edge, distance in zip(edgelist, distance)},
                                 font_color='red', font_size=9, font_family='Times New Roman')


def draw_route2(DR):
    pos = {n: [n.position[1], n.position[0]] for n in list(graph1.graph.nodes)}
    nx.draw(graph1.graph, pos, node_size=300)
    nx.draw_networkx_labels(graph1.graph, pos,
                            labels={n: n.name for n in list(graph1.graph.nodes)},
                            font_size=10, font_family='Times New Roman')
    fishground = list(filter(lambda node: isinstance(node, FishGround), list(graph1.graph.nodes())))
    stockPos = {n: [n.position[1] + 3, n.position[0] + 3] for n in list(fishground)}
    position = [[12, 35], [55, 25], [57, 62]]
    stockPos = {n: position[j] for n, j in zip(list(fishground), [0, 1, 2])}
    nx.draw_networkx_labels(graph1.graph, stockPos,
                            labels={n: 'Fishstock: {}'.format(n.fishstock['']['quantity']) for n in fishground},
                            font_color='green',
                            font_size=9, font_family='Times New Roman')
    edges_without_labels = [(loc2, ff2), (loc3, ff2)]
    edges_with_labels = list(filter(lambda edge: edge not in edges_without_labels, graph1.graph.edges))
    nx.draw_networkx_edge_labels(graph1.graph, pos,
                                 edge_labels={d: "D:{:.0f}".format(graph1.graph.edges[d]['distance']) for d in
                                              edges_with_labels},
                                 font_size=9, font_family='Times New Roman')
    width = [x / 5 for x in DR[1]]
    pos = {n: [n.position[1], n.position[0]] for n in list(graph1.graph.nodes())}
    nx.draw_networkx_nodes(G=graph1.graph, pos=pos, nodelist=DR[0], node_color='red', node_size=300)
    edgelist1 = [(DR[0][i], DR[0][i + 1]) for i in range(len(DR[0]) - 1)]
    edgelist = list(filter(lambda edge: edge not in edges_without_labels, edgelist1))
    nx.draw_networkx_edges(G=graph1.graph, pos=pos, edgelist=edgelist, edge_color='red', width=width)
    distance = [graph1.graph.edges[edge]['distance'] for edge in edgelist]
    nx.draw_networkx_edge_labels(G=graph1.graph,
                                 pos=pos,
                                 edge_labels={edge: "D:{:.0f}\nS:{:.0f}".format(distance, speed) for
                                              edge, distance, speed in zip(edgelist, distance, DR[1])},
                                 font_color='red', font_size=9, font_family='Times New Roman')
    nodePos = {n: [n.position[1] + 5, n.position[0] - 1] for n in DR[0]}
    nx.draw_networkx_labels(graph1.graph, nodePos,
                            labels={n: 'Du:{}'.format(d) for n, d in zip(DR[0][1:-1], DR[2])},
                            font_color='red', font_size=9, font_family='Times New Roman')


def draw_multi_route(data):
    for i in range(0, 5):
        plt.figure(figsize=(6, 4))
        plt.title('Route chosen by trained agent for %s' % (labels[i]), y=0.95)
        N = len(data)
        if i in [0, 1, 2]:
            draw_route1(data[i])
        else:
            draw_route2(data[i])
        plt.tight_layout()
        plt.savefig(fname='optimal_routes %s.png' % (i + 1))
        plt.show()


def draw_multi_route2(data):
    plt.figure(figsize=(6, 20))
    for i in range(0, 5):
        plt.subplot(5, 1, i + 1)
        plt.title('Route chosen by trained agent for %s' % (labels[i]), y=0.95)
        N = len(data)
        if i in [0, 1, 2]:
            draw_route1(data[i])
        else:
            draw_route2(data[i])
        plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.savefig(fname='optimal_routes.png')
    plt.show()


DR1 = [[start, loc1, ff2, end]]
DR2 = [[start, loc2, loc3, loc1, ff1, end]]
DR3 = [[start, loc2, loc3, ff2, end]]
DR4 = [[start, loc2, loc3, ff2, end], [19, 19, 19, 19], [1, 1, 1]]
DR5 = [[start, loc2, loc3, ff2, end], [19, 19, 19, 19], [1, 1, 1]]
data = [DR1, DR2, DR3, DR4, DR5]
#draw_multi_route(data)
# draw_multi_route2(data)
draw_fishing_network()
