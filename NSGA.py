"""
This file contains the functions to obtain Pareto optimal paths.
Made by Tiantian Zhu, tiantian.zhu@uit.no 2023/11/15
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pandas
#import pygad as gad
import numpy as np
import itertools
import random
from copy import deepcopy



def dominator(p, q, objectivefuncs):
    """
    To check whether p dominate q
    :param p: solution p
    :param q: solution q
    :param objectivefunc: objective functions to eveluate whether p dominate q
            format of objectivefuncs:
                objectivefuncs = {'minimize':[pathRisk, pathDistance], 'maximize':[pathCatch]}
            Need to specify the objective to be minimized and maximized.
    :return: True or False
    """
    "Code below are to calculate the value of all objectives for both p and q "
    p_outputs = [func(p) for func in objectivefuncs['maximize'] + objectivefuncs['minimize']]
    q_outputs = [func(q) for func in objectivefuncs['maximize'] + objectivefuncs['minimize']]
    objectiveNo = len(objectivefuncs['maximize'] + objectivefuncs['minimize'])  # get the number of objectives
    "For the first condition: p is not worse than q in all objectives"
    "Code below is to generate the comparison result regarding p is not worse than q in all objectives"
    compare_operator1 = lambda i: True if p_outputs[i] >= q_outputs[
        i] else False  # compare operator for objectives to be maximized.
    compare_operator2 = lambda i: True if p_outputs[i] <= q_outputs[
        i] else False  # compare operator for objectives to be minimized.
    compare_results1 = [compare_operator1(i) for i in range(0, len(objectivefuncs['maximize']))] + [compare_operator2(i)
                                                                                                    for i in range(
            len(objectivefuncs['maximize']), objectiveNo)]
    "For the second condition: p is strictly better than q at least in one objective"
    "Code below is to generate the comparison result regarding p is strictly better than q in all objectives"
    compare_operator3 = lambda i: True if p_outputs[i] > q_outputs[
        i] else False  # compare operator for objectives to be maximized.
    compare_operator4 = lambda i: True if p_outputs[i] < q_outputs[
        i] else False  # compare operator for objectives to be minimized.
    compare_results2 = [compare_operator3(i) for i in range(0, len(objectivefuncs['maximize']))] + [compare_operator4(i)
                                                                                                    for i in range(
            len(objectivefuncs['maximize']), objectiveNo)]
    if all(val is True for val in compare_results1):  # check whether p is not worse than q in all objectives
        if any(val is True for val in compare_results2):  # check whether p is strictly better than q in any objective
            return True  # both two conditions are met, so p dominate q, return True
    else:
        return False  # one of the two conditions are not met, so p does not dominate q, return False


def constraint_handling(paths, constraints, constraint_funcs):
    """
    To remove candidates outside a defined list of constraints
    and get the candidates that meet the constraints' requirements
    :parameter:
         1: paths: a list of all solutions
         2: constraints: a dictionary contains all constraints
             example the format of constraints:
                constraints = {'risk_constraint': {'low': 0, 'include_low': True, 'high': 400, 'include_high': True},
                               'distance_constraint': {'low': 0, 'include_low': True, 'high': 1900, 'include_high': True},
                               'catch_constraint': {'low': 0, 'include_low': True, 'high': 800, 'include_high': True}
                               }
                lowerbound and higherbound need to be specified. Whether lowerbound and higherbound are included in the contraints also needs to be specified.
                if include_low is True and include_high is True, it means the constraints function is: low <= Value <= high
                If include_low is False and include_high is False, it means the constraints function is: low < Value < high
                etc.
         3: constraint_funcs: a list of functions for constraints.
            The order for constraint functions should be the same as the order in constraints.
    :return: a new list of solution candidates which all meet the constraints
    """

    newpaths = []  # create an empty list to store solutions that meet all the constraints.
    #print('paths', paths)
    for path in paths:  # loop through all solutions
        withinconstraints = []  # an empty list to store whether the path meet each constraint
        values = [func(path) for func in constraint_funcs]  # calculate the objective values for all constraint
        # functions. The list of values are going to be used to compare with the lower bound and higher bound of each
        # constrant.
        for i, constraint in zip(range(0, len(constraints)), constraints):
            if constraints[constraint]['include_high']:
                if constraints[constraint]['include_low']:
                    if constraints[constraint]['low'] <= values[i] <= constraints[constraint]['high']:
                        withinconstraints.append(True)  # the objective value is within the constraint
                    else:
                        withinconstraints.append(False)  # the objective value is not within the constraint
                        break   # stop and go to next path.
                else:
                    if constraints[constraint]['low'] < values[i] <= constraints[constraint]['high']:
                        withinconstraints.append(True)
                    else:
                        withinconstraints.append(False)
                        break
            else:
                if constraints[constraint]['include_low']:
                    if constraints[constraint]['low'] <= values[i] < constraints[constraint]['high']:
                        withinconstraints.append(True)
                    else:
                        withinconstraints.append(False)
                        break
                else:
                    if constraints[constraint]['low'] < values[i] < constraints[constraint]['high']:
                        withinconstraints.append(True)
                    else:
                        withinconstraints.append(False)
                        break
        if all(val is True for val in withinconstraints):  # if the path is within all constraints.
            newpaths.append(path)
    return newpaths


def pareto_optima(paths, objectivefuncs):
    """
    To produce the ranks of all paths, pareto fronts, and pareto front that each solution belongs to.
    :param paths: A list of all solutions
    :param objectivefuncs: a list of objective functions which contains all objectives
    :return:
            a list of rank of all solutions according to the index of solutions. it looks like [1, 2, 3, 1]
            solutions in each pareto fronts from front 1 to the last front.
                it looks like: [[path1, path4], [path2, path3]]
            a list of pareto fronts for each solution according to the index of solutions. it looks like[1, 2, 2, 1]
    """
    "remove duplicates solutions"
    newpaths = list()
    for path in paths:
        if path not in newpaths:
            newpaths.append(path)
    paths = deepcopy(newpaths)
    F = [[]] * (len(paths)+2) # Create an empty list to store all fronts. each front contains one or more paths.
    S = [[] for i in range(0, len(paths))]  # create an empty list to store paths which are dominated by each path.
    N = [0] * len(paths)  # create an empty list to store the number of paths that dominate each path.
    Rank = ['nan'] * len(paths)  # create an empty list to store the rank of each path
    "Code below is to produce the first pareto front"
    F[1] = []  # initialize the first pareto front.
    for i in range(0, len(paths)):  # loop through all paths
        path_p, S[i], N[i] = paths[i], list(), 0
        pathscopy = deepcopy(paths)  # create a copy of paths
        pathscopy.remove(path_p)  # remove path from pathscopy so that path is not looped again.
        for path_q in pathscopy:  # look through all paths in pathscopy
            if dominator(path_p, path_q, objectivefuncs):  # get the set of paths p dominate.
                S[i].append(path_q)
            if dominator(path_q, path_p, objectivefuncs):  # get the Number of paths dominate p.
                N[i] += 1
        if N[i] == 0:  # if no path dominate path_p
            Rank[i] = 1  # assign rank 1 to path_p
            if path_p in F[1]:  # if path_p is already in the first pareto front
                continue
            else:  # add path_p to the first pareto front if it is not there.
                F[1].append(path_p)
    "Code below is to produce the rest of fronts"
    j = 1
    while len(F[j]) > 0:
        #print('front %s' %j, F[j])
        Q = []  # create an empty list to contain all paths that belong to the next front to be produced.
        for path_p in F[j]:  # loop through all paths in the current front.
            p_index = paths.index(path_p)
            for path_q in S[p_index]:  # loop through all paths that are dominated by path_p
                q_index = paths.index(path_q)
                N[q_index] = N[q_index] - 1
                # print(N[q_index])
                if N[q_index] == 0:
                    Rank[q_index] = j + 1
                    "Code below assigning path_q to the new front"
                    if path_q in Q:
                        continue
                    else:
                        Q.append(path_q)
                    newF = deepcopy(F[j])
                    newF.remove(path_p)  # remove path_p from F[j]
                    "code below is to remove path_q from all sets that dominated by a path, so that it will not be looped again."
                    for path in newF:
                        index = paths.index(path)
                        if path_q in S[index]:
                            S[index].remove(path_q)
        j += 1
        F[j] = Q
    F = [val for val in F if val != []]  # remove empty fronts.
    "Code below is to obtain the front No for each path"
    pathfront = []
    for path in paths:
        for front in F:
            if path in front:
                frontNo = F.index(front) + 1
                pathfront.append(frontNo)
    #print(['F%s %s %s:' % (j, len(F[j]), F[j]) for j in range(0, len(F))])
    #print('Fronts:', F)
    #print('Pareto front:', pathfront)
    #print('Rank:', Rank)
    #print('N:', N)
    return F, Rank, pathfront


def crowding_distance(paths, objectivefuncs):
    """
    To calculate the crowding distance for each path
    :param paths: a list of paths
    :param objectives: a list of objective functions
    :return: crowding distance of each path
    """
    l, M = len(paths), len(objectivefuncs['maximize'] + objectivefuncs['minimize'])
    allobjectivefuncs = objectivefuncs['maximize'] + objectivefuncs['minimize']
    path_distance, pathsinfo = [0] * l, pandas.DataFrame()
    pathsinfo['path'] = paths
    "Code below is to calculate value of each objective for each path and store the values in the dataframe pathsinfo"
    for m in range(0, M):
        pathsinfo[f'objective{m}'] = [allobjectivefuncs[m](path) for path in paths]
    "Code below is to calculate the crowding distance for each path."
    for i, path in zip(range(0, l), paths):  # loop through all paths.
        path_distance[i] = 0
        for m in range(0, M):  # calculate the crowding distance of each objective.
            objective = f'objective{m}'
            m_outputs = pathsinfo.sort_values(by=objective, ascending=True)
            m_outputs.reset_index()
            fm_min = m_outputs[objective].min()
            fm_max = m_outputs[objective].max()
            newlist = m_outputs['path'].tolist()
            path_index = newlist.index(path)
            if path_index in [0, l - 1]:  # if the path has the lowest or highest value in objective m.
                path_distance[i] = np.inf  # assign positive infinity to its crowding distance.
            else:
                pre = m_outputs.iloc[path_index + 1][objective]
                lat = m_outputs.iloc[path_index - 1][objective]
                path_dM = np.inf if fm_max == fm_min else (pre - lat) / (fm_max - fm_min)
                path_distance[i] += path_dM
    pathsinfo['crowding_distance'] = np.array(path_distance)
    return path_distance, pathsinfo


def nsga_optimal_route(paths, objectivefuncs, constraints, constraint_funcs, minimumDis, frontNo=1):
    """
    To obtain a list of paths which are optimal
    :param paths: a list of all paths
    :param objectivefuncs: objective functions to eveluate whether p dominate q
            format of objectivefuncs:
                objectivefuncs = {'minimize':[pathRisk, pathDistance], 'maximize':[pathCatch]}
            Need to specify the objective to be minimized and maximized.
    :param constraints: a dictionary contains all constraints
             example the format of constraints:
                constraints = {'risk_constraint': {'low': 0, 'include_low': True, 'high': 400, 'include_high': True},
                               'distance_constraint': {'low': 0, 'include_low': True, 'high': 1900, 'include_high': True},
                               'catch_constraint': {'low': 0, 'include_low': True, 'high': 800, 'include_high': True}
                               }
    :param constraint_funcs: a list of functions for constraints.
            The order for constraint functions should be the same as the order in constraints.
    :param minimumDis: The minimum crowding distance required.
    :param frontNo: The maximum pareto fronts to be included to select the optimal paths
    :return: a list of paths which are optimal
    """
    newpaths = constraint_handling(paths, constraints, constraint_funcs)
    fronts = pareto_optima(paths=newpaths, objectivefuncs=objectivefuncs)[2]
    distance = crowding_distance(newpaths, objectivefuncs)[0]
    optimal_path = []
    for i, path in zip(range(0, len(newpaths)), newpaths):
        if distance[i] >= minimumDis and fronts[i] <= frontNo:
            optimal_path.append(path)
    return optimal_path



def plot_pareto_optima_route(G, pos, optimal_routes):
    """
    To plot the calculated pareto optimal routes in a graph
    :param optimal_routes: a list of optimal routes
    """
    nx.draw(G, pos, edge_color='white', node_color='white', with_labels=True)
    colorlist = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'black', 'grey', 'olive',
                 'cornflowerblue'] * round(1 + (len(optimal_routes) / 10))
    for i in range(0, len(optimal_routes)):
        route = optimal_routes[i]
        rad = (i - len(optimal_routes) / 2) * 0.05
        edgelist = [(route[j], route[j + 1]) for j in range(0, len(route) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=1, edge_color=colorlist[i], node_size=600,
                               connectionstyle=f'arc3, rad = {rad}')
        nx.draw_networkx_nodes(G, pos, nodelist=route, edgecolors="tab:grey", node_color="tab:red",
                               node_size=i * 100, alpha=0.1)


def plot_sol_objective(paths_with_attributes, rank):
    """

    :param paths_with_attributes:
    :return:
    """
    plt.scatter(paths_with_attributes['total_gain'], paths_with_attributes['total_distance'])
    plt.xlabel('total_gain')
    plt.ylabel('total_distance')
    plt.grid()
    index = 0
    for x, y in zip(paths_with_attributes['total_gain'], paths_with_attributes['total_distance']):
        index += 1
        label = '%s(%s)' %(index, rank[index-1])
        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')


def constraint_handling_2(pathsinfo, constraints):
    """
    To remove candidates outside a defined list of constraints
    and get the candidates that meet the constraints' requirements
    :parameter:
         1: pathsinfo: a dataframe with paths and all objective values
         2: constraints: a dictionary contains all constraints
             example the format of constraints:
                constraints = {'total_risk': {'low': 0, 'include_low': True, 'high': 400, 'include_high': True},
                               'total_distance': {'low': 0, 'include_low': True, 'high': 1900, 'include_high': True},
                               'total_catch': {'low': 0, 'include_low': True, 'high': 800, 'include_high': True}
                               }
                lowerbound and higherbound need to be specified. Whether lowerbound and higherbound are included in the contraints also needs to be specified.
                if include_low is True and include_high is True, it means the constraints function is: low <= Value <= high
                If include_low is False and include_high is False, it means the constraints function is: low < Value < high
                etc.
         3: const_objectives: a list of objectives for the constraints.
            The name of each const objectives should be the same as the ones in constraints.
    :return: a new list of solution candidates which all meet the constraints
    """
    index = pathsinfo.index
    if len(index) == 0:
        return pathsinfo
    else:
        const_objectives = list(constraints.keys())
        for l in index:  # loop through all paths
            #print('index', l)
            #print(pathsinfo)
            withinconstraints = []
            values = pathsinfo.loc[l, const_objectives]
            #print('values', values)
            #print('values', values['total catch'])
            #print(constraints['total catch']['include_low'], constraints['total catch']['include_high'])
            # calculate the objective values for all constraint
            # functions. The list of values are going to be used to compare with the lower bound and higher bound of each
            # constraint.
            #print(values)
            for constraint in const_objectives:
                #print(constraint)
                if constraints[constraint]['include_high']:
                    if constraints[constraint]['include_low']:
                        #print('value of constraints', values[constraint])
                        #print('low', constraints[constraint]['low'])
                        #print('high', constraints[constraint]['high'])
                        if constraints[constraint]['low'] <= values[constraint] <= constraints[constraint]['high']:
                            withinconstraints.append(True)  # the objective value is within the constraint
                        else:
                            withinconstraints.append(False)  # the objective value is not within the constraint
                            break   # stop and go to next path.
                    else:
                        if constraints[constraint]['low'] < values[constraint] <= constraints[constraint]['high']:
                            withinconstraints.append(True)
                        else:
                            withinconstraints.append(False)
                            break
                else:
                    if constraints[constraint]['include_low']:
                        if constraints[constraint]['low'] <= values[constraint] < constraints[constraint]['high']:
                            withinconstraints.append(True)
                        else:
                            withinconstraints.append(False)
                            break
                    else:
                        if constraints[constraint]['low'] < values[constraint] < constraints[constraint]['high']:
                            withinconstraints.append(True)
                        else:
                            withinconstraints.append(False)
                            break
            #print(withinconstraints)
            if any(val is False for val in withinconstraints):  # if the path is within all constraints.
                pathsinfo = pathsinfo.drop(l, axis='index')
        pathsinfo = pathsinfo.reset_index(drop=True)
        return pathsinfo


def dominator_2(pathsinfo, p, q, objectives):
    """
    To check whether p dominate q

    :param pathsinfo: a dataframe includes all paths and objective values
    :param p: index number of solution p
    :param q: index number of solution q
    :param objectives: objectives to eveluate whether p dominate q
            format of objectives:
                objectives = {'minimize':['total_risk', 'total_distance'], 'maximize':['total_catch']}
            Need to specify the objective to be minimized and maximized.
    :return: True or False
    """
    "Code below are to calculate the value of all objectives for both p and q "
    objectiveNo = len(objectives['maximize'] + objectives['minimize'])  # get the number of objectives
    objectivelist = objectives['maximize'] + objectives['minimize']
    p_outputs = [pathsinfo.iloc[p][objectivelist[m]] for m in range(0, objectiveNo)]
    q_outputs = [pathsinfo.iloc[q][objectivelist[m]] for m in range(0, objectiveNo)]
    "For the first condition: p is not worse than q in all objectives"
    "Code below is to generate the comparison result regarding p is not worse than q in all objectives"
    compare_operator1 = lambda i: True if p_outputs[i] >= q_outputs[
        i] else False  # compare operator for objectives to be maximized.
    compare_operator2 = lambda i: True if p_outputs[i] <= q_outputs[
        i] else False  # compare operator for objectives to be minimized.
    compare_results1 = [compare_operator1(i) for i in range(0, len(objectives['maximize']))] + [compare_operator2(i)
                                                                                                    for i in range(
            len(objectives['maximize']), objectiveNo)]
    "For the second condition: p is strictly better than q at least in one objective"
    "Code below is to generate the comparison result regarding p is strictly better than q in all objectives"
    compare_operator3 = lambda i: True if p_outputs[i] > q_outputs[
        i] else False  # compare operator for objectives to be maximized.
    compare_operator4 = lambda i: True if p_outputs[i] < q_outputs[
        i] else False  # compare operator for objectives to be minimized.
    compare_results2 = [compare_operator3(i) for i in range(0, len(objectives['maximize']))] + [compare_operator4(i)
                                                                                                    for i in range(
            len(objectives['maximize']), objectiveNo)]
    if all(val is True for val in compare_results1):  # check whether p is not worse than q in all objectives
        if any(val is True for val in compare_results2):  # check whether p is strictly better than q in any objective
            return True  # both two conditions are met, so p dominate q, return True
    else:
        return False  # one of the two conditions are not met, so p does not dominate q, return False


def pareto_optima_2(pathsinfo, objectives):
    """
    To produce the ranks of all paths, pareto fronts, and pareto front that each solution belongs to.
    :param pathsinfo: A dataframe with all paths and objective values
    :param objectives: a list of objectives which contains all objectives to be included in the multi objective optimization
                        format:
                        objectives = {'minimize': ['total_risk', 'total_distance'], 'maximize': ['total_catch']}
    :return:
            1) solutions in each pareto fronts from front 1 to the last front.
                it looks like: [[0, 3], [1, 2]]
            2) updated pathsinfo with more information about the pareto front for each path, rank of each path.
    """
    try:
        pathsinfo = pathsinfo.drop('pathfront', axis='columns')
    except KeyError:
        pass
    pathsinfo = pathsinfo.reset_index(drop=True)
    #print('reindexed:', pathsinfo)
    F = [[]] * (len(pathsinfo)+2) # Create an empty list to store all fronts. each front contains one or more paths.
    S = [[] for i in range(0, len(pathsinfo))]  # create an empty list to store paths which are dominated each path.
    N = [0] * len(pathsinfo)  # create an empty list to store the number of paths that dominate each path.
    Rank = ['nan'] * len(pathsinfo)  # create an empty list to store the rank of each path
    "Code below is to produce the first pareto front"
    F[1] = []  # initialize the first pareto front.
    pathsindex = pathsinfo.index.tolist()
    for i in range(0, len(pathsinfo)):  # loop through all paths
        path_p, S[i], N[i] = pathsindex[i], list(), 0
        indexcopy = pathsindex.copy()  # create a copy of paths
        indexcopy.remove(path_p)  # remove path from pathscopy so that path is not looped again.
        for path_q in indexcopy:  # look through all paths in pathscopy
            if dominator_2(pathsinfo, path_p, path_q, objectives):  # get the set of paths p dominate.
                S[i].append(path_q)
            if dominator_2(pathsinfo, path_q, path_p, objectives):  # get the Number of paths dominate p.
                N[i] += 1
        if N[i] == 0:  # if no path dominate path_p
            Rank[i] = 1  # assign rank 1 to path_p
            if path_p in F[1]:  # if path_p is already in the first pareto front
                continue
            else:  # add path_p to the first pareto front if it is not there.
                F[1].append(path_p)
    "Code below is to produce the rest of fronts"
    j = 1
    while len(F[j]) > 0:
        Q = [] # create an empty list to contain all paths that belong to the next front to be produced.
        for path_p in F[j]:  # loop through all paths in the current front.
            p_index = pathsindex.index(path_p)
            for path_q in S[p_index]:  # loop through all paths that dominated by path_p
                q_index = pathsindex.index(path_q)
                N[q_index] = N[q_index] - 1
                # print(N[q_index])
                if N[q_index] == 0:
                    Rank[q_index] = j + 1
                    "Code below assigning path_q to the new front"
                    if path_q in Q:
                        continue
                    else:
                        Q.append(path_q)
                    newF = F[j].copy()
                    newF.remove(path_p)  # remove path_p from F[j]
                    "Code below is to remove path_q from all sets that dominated by a path, so that it will not be " \
                    "looped again."
                    for path in newF:
                        index = pathsindex.index(path)
                        if path_q in S[index]:
                            S[index].remove(path_q)
        j += 1
        F[j] = Q
    F = [val for val in F if val != []]  # remove empty fronts.
    "Code below is to obtain the front No for each path"
    pathfront = []
    for path in pathsindex:
        for front in F:
            if path in front:
                frontNo = F.index(front) + 1
                pathfront.append(frontNo)
    # print(['F%s %s %s:' % (j, len(F[j]), F[j]) for j in range(0, len(F))])
    # print('Fronts:', F)
    # print('Pareto front:', pathfront)
    # print('Rank:', Rank)
    # print('N:', N)
    pathsinfo['pathfront'] = pathfront
    pathsinfo['rank'] = Rank
    return F, pathsinfo


def crowding_distance_2(pathsinfo, objectives=None):
    """
    To calculate the crowding distance for each path
    :param pathsinfo: a dataframe with paths and objective values
    :param objectives: objectives to be optimized. format:objectives = {'minimize': ['total_risk', 'total_distance'], 'maximize': ['total_catch']}
    :return: an updated pathsinfo with crowding distance for each solution.
    """
    pathsinfo = pathsinfo.copy()
    try:
        pathsinfo = pathsinfo.drop('distance', axis='columns')
    except KeyError:
        pass
    try:
        objectives = objectives['minimize'] + objectives['maximize']
    except KeyError:
        print('Objectives are required')
    M = len(objectives)
    l = len(pathsinfo)
    paths = pathsinfo['path'].tolist()
    path_distance = [0] * l
    for i, path in zip(range(0, l), paths):
        path_distance[i] = 0
        for m in range(0, M):
            m_outputs = pathsinfo[['path', objectives[m]]]
            m_outputs = m_outputs.sort_values(by=objectives[m], ascending=True)
            m_outputs = m_outputs.reset_index()
            fm_min = m_outputs[objectives[m]].min()
            fm_max = m_outputs[objectives[m]].max()
            newlist = m_outputs['path'].tolist()
            path_index = newlist.index(path)
            if path_index in [0, l - 1]:
                path_distance[i] = np.inf
            else:
                pre = m_outputs.iloc[path_index + 1][objectives[m]]
                lat = m_outputs.iloc[path_index - 1][objectives[m]]
                path_dM = np.inf if fm_max == fm_min else (pre-lat) / (fm_max - fm_min)
                path_distance[i] += path_dM
                #print('distance', path_distance[i])
    #pathsinfo['distance'] = path_distance
    pathsinfo['crowding_distance'] = path_distance
    #print(len(pathsinfo), len(path_distance))
    return pathsinfo


def nsga_optimal_route_2(pathsinfo, objectives, constraints, minimumDis, frontNo=1):
    """
    To get the
    :param pathsinfo: a DataFrame with all paths and objectives values and constraints values
    :param objectives: objectives in a format:{'minimize': ['total_risk', 'total_distance'], 'maximize': ['total_catch']}
    :param constraints: a dictionary contains all constraints
             example the format of constraints:
                constraints = {'total_risk': {'low': 0, 'include_low': True, 'high': 400, 'include_high': True},
                               'total_distance': {'low': 0, 'include_low': True, 'high': 1900, 'include_high': True},
                               'total_catch': {'low': 0, 'include_low': True, 'high': 800, 'include_high': True}
                               }
    :param const_objectives: a list of objectives for the constraints.
            The name of each const objectives should be the same as the ones in constraints.
    :param minimumDis: minimum crowding distance requirement
    :param frontNo: pareto fronts to be included in the set of optimal paths
    :return: a new dataframe which contains the obtained optimal paths
    """
    newpathsinfo = constraint_handling_2(pathsinfo, constraints, list(constraints.keys()))
    newpathsinfo = pareto_optima_2(pathsinfo=newpathsinfo, objective=objectives)[1]
    newpathsinfo = crowding_distance_2(pathsinfo=newpathsinfo, objectives=objectives)
    for i in range(0, len(newpathsinfo)):
        if newpathsinfo.loc[i, 'crowding_distance'] < minimumDis and newpathsinfo.loc[i, 'pathfront'] > frontNo:
            newpathsinfo.remove(newpathsinfo.iloc[i])
    return newpathsinfo

def generate_initial_solutions(gene_space, route, num):
    """
    To generate defined num of initial solutions which will be used for evolution.
    :param gene_space: gene_space for solution. gene_space is a list of ranges.
    :param route: route, the length of speed and duration is dependent on
             the length of route. Therefore, route information should be provided here.

     :param num: Number of solutions to be generated
     :return: initial solutions.
    """
    speed_space = gene_space[0]
    duration_space = gene_space[1]
    speed_matrix=[]
    duration_matrix=[]
    for n in range(num):
        speed_matrix.append(random.sample(speed_space, k=(len(route)-1)))
        duration_matrix.append(random.sample(duration_space, k=len(route)-2))
    initial_solutions = [[speed_matrix[i], duration_matrix[i]] for i in range(0, num)]
    return initial_solutions


def rank_solutions(solutions, objective_funcs, constraints, constraint_funcs, num):
    """
    To get a list of top num of solution in a sorted order in descending order.
    :param solutions: a list of solutions
    :param objective_funcs: objective functions use to sort solutions
    :param num: required number of solutions
    :return: a list of ranked required number of solutions.
    """
    if len(constraints) != 0:
        solutions = constraint_handling(solutions, constraints, constraint_funcs)
    Pareto = pareto_optima(solutions, objective_funcs)
    #print('number of solutions', len(solutions))
    F = Pareto[0]
    #print('pareto front:', F)
    sorted_solutions = []
    #num = min(len(F[0]), num)
    for f in F:
        #print('one pareto front:', f)
        length = len(sorted_solutions)
        length_f = len(f)
        if length <= num:
            m = num-length
            n = m if m <= length_f else length_f
            distance = crowding_distance(f, objective_funcs)[1]
            distance = distance.sort_values(by=['crowding_distance'], ascending=False)
            sorted_f = distance['path'].tolist()
            sorted_solutions += (sorted_f[0:n])
        else:
            break
    #print('sorted', sorted_solutions)
    return sorted_solutions


def rank_solutions_2(solutions, objective_funcs, constraints:{}, constraint_funcs:None, num):
    """
    To get a list of top num of solution in a sorted order in descending order.
    :param solutions: a list of solutions
    :param objective_funcs: objective functions use to sort solutions
    :param num: required number of solutions
    :return: a list of ranked required number of solutions.
    """
    minimizationObj = ['minimizationObj %s' % i for i in range(0, len(objective_funcs['minimize']))]
    maximizationObj = ['maximizationObj %s' % i for i in range(0, len(objective_funcs['maximize']))]
    objectives = {'minimize': minimizationObj, 'maximize': maximizationObj}
    constraints_name = list(constraints.keys())
    try:
        solutionsinfo = pandas.DataFrame(columns=['path'] + minimizationObj + maximizationObj + constraints_name)
        solutionsinfo['path'] = solutions
        for s, i in zip(constraints_name, range(0, len(constraints_name))):
            solutionsinfo[s] = [constraint_funcs[i](solution) for solution in solutions]
    except KeyError:
        solutionsinfo = pandas.DataFrame(columns=['path'] + minimizationObj + maximizationObj)
        solutionsinfo['path'] = solutions
    for s, i in zip(minimizationObj, range(0, len(minimizationObj))):
        solutionsinfo[s] = [objective_funcs['minimize'][i](solution) for solution in solutions]
    for s, i in zip(maximizationObj, range(0, len(maximizationObj))):
        solutionsinfo[s] = [objective_funcs['maximize'][i](solution) for solution in solutions]
    if len(constraints) != 0:
        solutionsinfo = constraint_handling_2(solutionsinfo, constraints)
        #print('dataframe after constraints handling', solutionsinfo)
        if len(solutionsinfo) == 0:
            return pandas.DataFrame(), []
    Pareto = pareto_optima_2(solutionsinfo, objectives=objectives)
    #print('number of solutions', len(solutions))
    F = Pareto[0]
    #print('pareto fronts:', F)
    solutionsinfo = Pareto[1]
    #print('pareto front:', F)
    sorted_solutions = []
    for f in F:
        f = [solutionsinfo['path'][i] for i in f]
        #print('f', f)
        length = len(sorted_solutions)
        length_f = len(f)
        if length < num:
            m = num - length
            n = m if m <= length_f else length_f
            distance = crowding_distance(f, objective_funcs)[1]
            distance = distance.sort_values(by=['crowding_distance'], ascending=False)
            sorted_f = distance['path'].tolist()
            sorted_solutions += (sorted_f[0:n])
        else:
            break
    #print('sorted solutions', sorted_solutions)
    return sorted_solutions


def rank_solutions_3(solutions, objective_funcs, constraints:{}, constraint_funcs:None, num):
    """
    To get a list of top num of solution in a sorted order in descending order.
    :param solutions: a list of solutions
    :param objective_funcs: objective functions use to sort solutions
    :param num: required number of solutions
    :return: a list of ranked required number of solutions.
    """
    minimizationObj = ['minimizationObj %s' % i for i in range(0, len(objective_funcs['minimize']))]
    maximizationObj = ['maximizationObj %s' % i for i in range(0, len(objective_funcs['maximize']))]
    objectives = {'minimize': minimizationObj, 'maximize': maximizationObj}
    try:
        constraints_name = list(constraints.keys())
        solutionsinfo = pandas.DataFrame(columns=['path'] + minimizationObj + maximizationObj + constraints_name)
        solutionsinfo['path'] = solutions
        for s, i in zip(constraints_name, range(0, len(constraints_name))):
            solutionsinfo[s] = [constraint_funcs[i](solution) for solution in solutions]
    except KeyError:
        solutionsinfo = pandas.DataFrame(columns=['path'] + minimizationObj + maximizationObj)
        solutionsinfo['path'] = solutions
    for s, i in zip(minimizationObj, range(0, len(minimizationObj))):
        solutionsinfo[s] = [objective_funcs['minimize'][i](solution) for solution in solutions]
    for s, i in zip(maximizationObj, range(0, len(maximizationObj))):
        solutionsinfo[s] = [objective_funcs['maximize'][i](solution) for solution in solutions]
    if len(constraints) != 0:
        constraints_name = list(constraints.keys())
        solutionsinfo = constraint_handling_2(solutionsinfo, constraints)
        #print('dataframe after constraints handling', solutionsinfo)
        if solutionsinfo.empty:
            return pandas.DataFrame(), []
    if len(minimizationObj+maximizationObj) != 0:
        Pareto = pareto_optima_2(solutionsinfo, objectives=objectives)
        #print('number of solutions', len(solutions))
        F = Pareto[0]
        solutionsinfo = Pareto[1]
        #print('pareto front:', F)
        sorted_solutionsinfo = pandas.DataFrame()
        for k in range(1, len(F)+1):
            f = solutionsinfo[solutionsinfo['pathfront']==k]
            #print('path in a front', f)
            length = len(sorted_solutionsinfo)
            length_f = len(f)
            if length <= num:
                m = num - length
                n = m if m <= length_f else length_f
                f_with_distance_value = crowding_distance_2(f, objectives)
                f_with_distance_value = f_with_distance_value.sort_values(by='crowding_distance', ascending=False)
                #print('sorted by distance', f_with_distance_value['distance'])
                sorted_solutionsinfo = pandas.concat([sorted_solutionsinfo, f_with_distance_value.head(n)])
            else:
                break
        sorted_solutions = sorted_solutionsinfo['path'].tolist()
        #print('sorted', sorted_solutions)
        return sorted_solutionsinfo, sorted_solutions
    else:
        return solutionsinfo, solutionsinfo['path'].tolist()


def selection_pair(parents):
    """
    select a pair of parents for crossover operation
    :param parents: a list of parents
    :return: two parents will be used for mutation
    """
    pairs = itertools.combinations(parents, 2)
    new_pairs = []
    for pair in pairs:
        element1 = pair[0]
        element2 = pair[1]
        el = [element1, element2]
        new_pairs.append(el)
    return new_pairs

def crossover_operation(pair):
    """

    :return:
    """

    a = pair[0]
    b = pair[1]
    if len(a) != len(b):
        raise ValueError('parents must be of the same length')
    length = len(a)
    if len(a) < 2:
        return pair
    p = random.randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def crossover_operator_2(pair):
    lst_a = pair[0]
    lst_b = pair[1]
    child_pair = [[], []]
    for subpair in zip(lst_a, lst_b):
        a = subpair[0]
        b = subpair[1]
        if len(a) != len(b):
            raise ValueError('parents must be of the same length')
        length = len(a)
        if len(a) < 2:
            child_subpair = list(subpair)
        else:
            p = random.randint(1, length - 1)
            child_subpair = [a[0:p] + b[p:], b[0:p] + a[p:]]
        child_pair[0].append(child_subpair[0])
        child_pair[1].append(child_subpair[1])
    return child_pair

def mutation_operator(child, gene_space, num:int=1, probability: float = 0.5):
    """

    :return:
    """
    for _ in range(0, num):
        index = random.randrange(0, len(child))
        child[index] = child[index] if random.random() > probability else random.randrange(gene_space)
    return child

def mutation_operator_2(child, gene_space, n: int = 1, probability: float = 0.5):
    """
    Method for mutation
    :param child: Child solution generated from crossover operator
    :param gene_space:
    :param num:
    :param probability:
    :return:
    """
    length = len(child)
    #print('child', child)
    for i, parameter in zip(range(0, length), child):
        if len(parameter) == 0:
            child[i] = parameter
        else:
            for j in range(0, n):
                index = random.randrange(0, len(parameter))
                random.seed()
                newValue = random.choice(gene_space[i])
                random.seed()
                parameter[index] = parameter[index] if random.random() > probability else newValue
                child[i] = parameter
    return child


def run_evolution(initial_solutions,
                  objective_funcs,
                  constraints,
                  constraint_funcs,
                  generation_limit, gene_space, num: int = 5, objectives_limit=None):
    """
    The main loop of evolutionary algorithm
    :param initial_solutions: Initial solutions to start.
    :param objective_funcs: Objective functions to evaluate the solutions
    :param objectives_limit: goals to be achieved for all objectives
    :param generation_limit: Planned number of generations
    :param gene_space: gene space for each parameter of solution
    :param num: Number of solutions to include for evolution
    :return: a list of optimal solutions.
    """
    parents = initial_solutions
    colors = range(0, generation_limit)
    green = plt.get_cmap('Greens')
    red = plt.get_cmap('Reds')
    nipy = plt.get_cmap('nipy_spectral')
    for i in range(generation_limit):
        parents = rank_solutions(parents, objective_funcs, constraints, constraint_funcs, num=num)
        if len(parents) == 0:
            parents = initial_solutions
        elif len(parents) <= 1:
            break
        #risk = [objective_funcs['maximize'][0](parents[i]) for i in range(0, len(parents))]
        #fuel = [objective_funcs['maximize'][1](parents[i]) for i in range(0, len(parents))]
        #plt.scatter(risk, fuel, color=green(i*5), marker="s", label='generation %s' % i)
        #print("generation %s" % i, parents)
        pairs = selection_pair(parents)
        pairs = random.choices(pairs, k=10)
        next_generation = []
        for pair in pairs:
            offsprings = crossover_operator_2(pair)
            #print('crossover', offsprings)
            offspring_a, offspring_b = mutation_operator_2(offsprings[0], gene_space), mutation_operator_2(offsprings[1], gene_space)
            next_generation.append(offspring_a)
            next_generation.append(offspring_b)
            #print('mutation', next_generation)
        parents = parents + next_generation
        #risk = [objective_funcs['minimize'][0](parents[i]) for i in range(0, len(parents))]
        #fuel = [objective_funcs['maximize'][0](parents[i]) for i in range(0, len(parents))]

        #plt.scatter(risk, fuel, color=red(i*10), marker="*", label='generation %s' % i)
        #print('all', parents)
    #plt.legend(fontsize=5)
    optimal_solutions = rank_solutions(parents, objective_funcs, num=num)
    return optimal_solutions

def run_evolution_2(initial_solutions,
                  objective_funcs,
                  constraints,
                  constraint_funcs,
                  generation_limit, gene_space, num: int = 5, objectives_limit=None):
    """
    The main loop of evolutionary algorithm
    :param initial_solutions: Initial solutions to start.
    :param objective_funcs: Objective functions to evaluate the solutions
    :param objectives_limit: goals to be achieved for all objectives
    :param generation_limit: Planned number of generations
    :param gene_space: gene space for each parameter of solution
    :param num: Number of solutions to include for evolution
    :return: a list of optimal solutions.
    """
    newparents = initial_solutions
    colors = range(0, generation_limit)
    green = plt.get_cmap('Greens')
    red = plt.get_cmap('Reds')
    nipy = plt.get_cmap('nipy_spectral')
    generationNo = 0
    while generationNo <= generation_limit:
        parents = rank_solutions(newparents, objective_funcs, constraints, constraint_funcs, num=num)
        parents_2 = deepcopy(parents)
        #print('parents after rank solution,', parents)
        if len(parents_2) == 0:
            parents_2 = deepcopy(initial_solutions)
        elif len(parents_2) <= 1:
            parents_2.extend(initial_solutions)
            #parents += initial_solutions
        #risk = [objective_funcs['maximize'][0](parents[i]) for i in range(0, len(parents))]
        #fuel = [objective_funcs['maximize'][1](parents[i]) for i in range(0, len(parents))]
        #plt.scatter(risk, fuel, color=green(i*5), marker="s", label='generation %s' % i)
        #print("generation %s" % i, parents)
        pairs = selection_pair(parents_2)
        random.seed()
        pairs = random.choices(pairs, k=6 if len(pairs) >= 6 else len(pairs))
        next_generation = []
        for pair in pairs:
            next_generation = deepcopy(next_generation)
            offsprings = crossover_operator_2(pair)
            offspring_a = mutation_operator_2(child=offsprings[0], gene_space=gene_space)
            offspring_b = mutation_operator_2(child=offsprings[1], gene_space=gene_space)
            next_generation.extend([offspring_a, offspring_b])
        #risk = [objective_funcs['minimize'][0](parents[i]) for i in range(0, len(parents))]
        #fuel = [objective_funcs['maximize'][0](parents[i]) for i in range(0, len(parents))]

        #plt.scatter(risk, fuel, color=red(i*10), marker="*", label='generation %s' % i)
        #print('all', parents)
        #plt.legend(fontsize=5)
        generationNo += 1
        newparents = parents + next_generation
        "code below is to remove duplicated solutions"
        newpaths = list()
        for path in newparents:
            if path not in newpaths:
                newpaths.append(path)
        newparents = deepcopy(newpaths)
    if len(newparents) == 0:
        optimal_solutions = [pandas.DataFrame(), []]
    else:
        optimal_solutions = rank_solutions_3(newparents, objective_funcs, constraints, constraint_funcs, num=num)
    #print('after evolution for each route', optimal_solutions)
    return optimal_solutions[0]






