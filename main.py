#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import networkx as nx
import numpy as np
from collections import deque


# This class is going to represent a graph, so that we can organize all the implementation and methods of
# our data structure in an organized way
class Graph:
    
    # This constructur takes as parameter the number of nodes
    def __init__(self, size):
        self._nodes = [[] for _ in range(size)]
        self._nodes_features = [{} for _ in range(size)]
        
    # This method adds features to a node in the form of a dictionary
    def add_features_to_node(self, node, features, replace = False):
        if(replace):
            self._nodes_features[node - 1] = features
        else:
            for key in features.keys():
                self._nodes_features[node - 1][key] = features[key]
                
    # This method is for removing all given features from the nodes of the graph, or all features if no second
    # parameter is given
    def reset_features(self, features = None):
        if(features is None):
            for index in range(len(self._nodes_features)):
                self._nodes_features[index] = {}
        else:
            for index in range(len(self._nodes_features)):
                for feature in features:
                    self._nodes_features[index].pop(feature, None)
        
    # This method is for getting features of a node
    def get_feature_of_node(self, node, feature):
        moment = self._nodes_features[node - 1]
        if(feature in moment.keys()):
            return(moment[feature])
        return(None)
        
    # This method adds an edge to the graph, together with an optional dict of weights
    # The edge is given as a tuple of nodes
    def add_edge(self, edge, weights = None, check = None):
        if(check):
            for neighbour in self.get_neighbours(edge[0]):
                if(neighbour == edge[1]):
                    return(None)
        to_add = (edge[1] - 1, {})
        if(not weights is None):
            to_add = (edge[1] - 1, weights)
        self._nodes[edge[0] - 1].append(to_add)
        
        to_add = (edge[0] - 1, {})
        if(not weights is None):
            to_add = (edge[0] - 1, weights)
        self._nodes[edge[1] - 1].append(to_add)
        
    # This method gets all neighbours of a node and given weights of the edges connecting them to the node
    def get_neighbours(self, node, features = None):
        if(features is None):
            return([single_node[0] + 1 for single_node in self._nodes[node - 1]])
        to_return = []
        for neighbour in self._nodes[node - 1]:
            to_return.append((neighbour[0] + 1, dict((k, neighbour[1][k]) for k in features if k in neighbour[1])))
        return(to_return)
    
    # This method get an edge from the graph together with the specified weights
    # If the edge doesn't exist it returns None
    def get_edge(self, node_one, node_two, features = None):
        for neighbour in self._nodes[node_one - 1]:
            if(neighbour[0] == node_two - 1):
                if(features is None):
                    return((node_one, node_two))
                return((node_one, node_two, dict((k, neighbour[1][k]) for k in features if k in neighbour[1])))
        return(None)
    
    # This method updates and edge adding more weights or replacing them (based on replace parameter)
    # It doesn't do anything if the edge doesn't exist
    def update_edge(self, node_one, node_two, weights):
        for neighbour in self._nodes[node_one - 1]:
            if(neighbour[0] == node_two - 1):
                for key in weights.keys():
                    neighbour[1][key] = weights[key]
                return
            
    # This method creates and returns the subgraph of this subgraph containing all the nodes in set_of_nodes
    # and all the edges connecting them
    # with_reference parameter estabilishes if this subgraph should have reference to nodes of starting graph
    # as features
    # features parameter estabilishes what edges features should be mantained
    def get_complete_subgraph(self, set_of_nodes, features, with_reference = False):
        to_return = Graph(len(set_of_nodes))
        list_set_of_nodes = [0]
        list_set_of_nodes.extend(list(set_of_nodes))
        for index in range(len(set_of_nodes)):
            for neighbour in self.get_neighbours(list_set_of_nodes[index + 1], features):
                if(neighbour[0] in set_of_nodes):
                    to_return.add_edge((index + 1, list_set_of_nodes.index(neighbour[0])), neighbour[1], check = True)
        if(with_reference):
            for index in range(len(set_of_nodes)):
                to_return.add_features_to_node(index + 1, {"Reference node" : list_set_of_nodes[index + 1]})
        return(to_return)
        
            
    # This method prints the adjacency list of the graph, it will be used just to check that everything works fine
    def print_graph(self, head = None):
        if(head is None):
            head = len(self._nodes)
        for index in range(head):
            print(index + 1, end = " : ")
            print([(node[0] + 1, node[1]) for node in self._nodes[index]])
            
            
            
counter = 0
nodes = open("/Users/digitalfirst/Desktop/adm_hw5/USA-road-d.CAL.co" , "r")
while(True):
    string = nodes.readline()
    if(string == ""):
        break
    if(string[0] == 'v'):
        counter = counter + 1
nodes.close()

CAL = Graph(counter)

times = open("/Users/digitalfirst/Desktop/adm_hw5/USA-road-t.CAL.gr", "r")
distances = open("/Users/digitalfirst/Desktop/adm_hw5/USA-road-d.CAL.gr", "r")
counter = 0
while(True):
    string_time = times.readline()
    string_distance = distances.readline()
    if(string_time == ""):
        break
    if(string_time[0] == 'a' and (counter % 2) == 1):
        edge_time = string_time.split(" ")
        edge_distance = string_distance.split(" ")
        CAL.add_edge((int(edge_time[1]), int(edge_time[2])), {"Time" : int(edge_time[3]), "Distance" : int(edge_distance[3])})  
    counter = counter + 1
times.close()
distances.close()


# This method finds the shortest path connecting a source and a target node using a breadth-first-search
# It returns None if not such a path exists
def breadth_first_search(self, source, target):
    in_queue_nodes = deque()
    
    in_queue_nodes.appendleft(source)
    self.add_features_to_node(source, {"Visited" : True})
    while(True):
        if(not in_queue_nodes):
            self.reset_features(["Visited", "Parent"])
            return(None)
        visiting_now = in_queue_nodes.pop()
        if(visiting_now == target):
            break
        for neighbour in self.get_neighbours(visiting_now):
            if(self.get_feature_of_node(neighbour, "Visited") is None):
                in_queue_nodes.appendleft(neighbour)
                self.add_features_to_node(neighbour, {"Parent" : visiting_now, "Visited" : True})
    
    to_return = []
    while(not visiting_now is None):
        to_return.append(visiting_now)
        visiting_now = self.get_feature_of_node(visiting_now, "Parent")
    self.reset_features(["Visited", "Parent"])
    
    return(to_return[::-1])

setattr(Graph, 'breadth_first_search', breadth_first_search)


# This class represents a binary heap using a list.
class Binary_Heap:
    
    # This constructor is for creating a binary heap containing a single node
    def __init__(self, single_node, int_function = lambda x : x):
        self._heap = [None, single_node]
        self._int_function = int_function
    
    # This constructor is for getting left child of a node given his index, None if not such a child exists
    def get_left_child(self, node_index):
        position = node_index * 2
        if(position >= len(self._heap)):
            return(None)
        return(self._heap[position])
    
    # Like left child, but for right child
    def get_right_child(self, node_index):
        position = (node_index * 2) + 1
        if(position >= len(self._heap)):
            return(None)
        return(self._heap[position])
    
    # This constructor is for getting parent of a node given is index, None if not such a parent exists
    def get_parent(self, node_index):
        return(self._heap[node_index // 2])
    
    # This is the heapify function, it helps preserving the properties of the heap
    def heapify(self, index):
        left = 2 * index
        right = (2 * index) + 1
        minimum = index
        if(left < len(self._heap) and self._int_function(self._heap[left]) < self._int_function(self._heap[minimum])):
            minimum = left
        if(right < len(self._heap) and self._int_function(self._heap[right]) < self._int_function(self._heap[minimum])):
            minimum = right
        if(minimum != index):
            aux = self._heap[index]
            self._heap[index] = self._heap[minimum]
            self._heap[minimum] = aux
            self.heapify(minimum)
        
    # This function is for inserting a new node inside the heap
    def insert(self, new_node):
        if(len(self._heap) == 1):
            self._heap.append(new_node)
            return
        self._heap.append(new_node)
        index = len(self._heap) - 1
        while(self._int_function(self._heap[index]) < self._int_function(self.get_parent(index))):
            aux = self._heap[index]
            self._heap[index] = self.get_parent(index)
            self._heap[index // 2] = aux
            index = index // 2
            if(index // 2 == 0):
                return
    
    # This function is for extracting the minimum element from the heap
    def extract(self):
        if(len(self._heap) == 2):
            return(self._heap.pop())
        to_return = self._heap[1]
        self._heap[1] = self._heap.pop()
        self.heapify(1)
        return(to_return)
    
    # Method for checking if the heap is empty
    def is_empty(self):
        return(len(self._heap) == 1)
    
# Dijkstra's algorithm, it finds the shortest path in a graph connecting source and target taking into consideration
# the given weight.
def dijkstra(self, source, target, weight):
    heap = Binary_Heap((source, 0), lambda x : x[1])
    self.add_features_to_node(source, {"Distance" : 0})
    while(True):
        if(heap.is_empty()):
            self.reset_features(["Visited", "Distance", "Parent"])
            return(None)
        current_node = heap.extract()
        if(current_node[0] == target):
            break
        if(self.get_feature_of_node(current_node[0], "Visited")):
            continue
        this_distance = self.get_feature_of_node(current_node[0], "Distance")
        for close_node in self.get_neighbours(current_node[0], [weight]):
            prev_distance = self.get_feature_of_node(close_node[0], "Distance")
            arc_weight = close_node[1][weight]
            new_distance = this_distance + arc_weight
            if(prev_distance is None or prev_distance > new_distance):
                self.add_features_to_node(close_node[0], {"Distance" : new_distance, "Parent" : current_node[0]})
                heap.insert((close_node[0], new_distance))
        self.add_features_to_node(current_node[0], {"Visited" : True})
    
    current_node = current_node[0]
    to_return = []
    while(not current_node is None):
        to_return.append(current_node)
        current_node = self.get_feature_of_node(current_node, "Parent")
    self.reset_features(["Visited", "Distance", "Parent"])
    return(to_return[::-1])

setattr(Graph, 'dijkstra', dijkstra)


def getNeigh(v,nodeList,d,typeDistance):
    #add current node in nodeList 
    if v not in nodeList:
        nodeList.append(v)
    #case of typeDistance=NetworkDistance    
    if typeDistance=="NetworkDistance":
        #get neighbours of current node v
        neighbours=CAL.get_neighbours(v)
        #iterate all neighbours
        for i in range(len(CAL.get_neighbours(v))):
            #get edge betwwen node v and current neighbours
            edge=CAL.get_edge(v,neighbours[i])
            #check if we can put this neighbours in nodeList
            if not d==0 and neighbours[i] not in nodeList:
                
                nodeList.append(neighbours[i])
                
                getNeigh(neighbours[i],nodeList,d-1,typeDistance)
        
        return nodeList
        
    else:
        #case of typeDistance is not NetworkDistance, i.e. it is 'Distance' or 'Time'
        
        #get neighbours with its information(time or distance) of current node v
        neighbours=CAL.get_neighbours(v,[typeDistance])
        
        #iterate all neighbours of node v
        for i in range(len(CAL.get_neighbours(v,[typeDistance]))):
            
            #get edge between node v and current its neighbour
            edge=CAL.get_edge(v,neighbours[i][0],[typeDistance])
            
            #check if we can put current neighbour in nodeList
            if edge[2][typeDistance] < d and neighbours[i][0] not in nodeList:
             
                nodeList.append(neighbours[i][0])
                
                getNeigh(neighbours[i][0],nodeList,d-edge[2][typeDistance],typeDistance)
        
        
        return nodeList

    
# This functin applies Prim's algorithm to a graph for finding minimum spanning tree
# weight parameter estabilishes which weight to use to choose the minimum tree
def prim_algorithm(self, weight = None):
    set_of_roads = set()
    heap = Binary_Heap((1, 0), lambda x : x[1])
    while(not heap.is_empty()):
        current_node = heap.extract()
        if(self.get_feature_of_node(current_node[0], "In tree")):
            continue
        self.add_features_to_node(current_node[0], {"In tree" : True})
        other_node = self.get_feature_of_node(current_node[0], "Chosen edge")
        set_of_roads.add(other_node)
        if(other_node):
            self.add_features_to_node(other_node[1], {"In tree" : True})
        for neighbour in self.get_neighbours(current_node[0], [weight]):
            if(self.get_feature_of_node(neighbour[0], "In tree")):
                continue
            if(weight):
                distance = neighbour[1][weight]
            else:
                distance = 1
            prev_distance = self.get_feature_of_node(neighbour[0], "Distance")
            if(prev_distance is None or prev_distance > distance):
                self.add_features_to_node(neighbour[0], {"Distance" : distance, "Chosen edge" : (neighbour[0], current_node[0])})
                heap.insert((neighbour[0], distance))
    self.reset_features(["Distance", "Chosen edge", "In tree"])
    return(set_of_roads)

setattr(Graph, 'prim_algorithm', prim_algorithm)

# This function applied Prim's algorithm to a complete subgraph of the graph
def smartest_network(self, set_of_nodes, weight = None):
    subgraph = self.get_complete_subgraph(set_of_nodes, [weight], with_reference = True)
    set_of_edges = subgraph.prim_algorithm(weight)
    to_return = set()
    for edge in set_of_edges:
        if(edge is None):
            continue
        start = edge[0]
        end = edge[1]
        real_start = subgraph.get_feature_of_node(start, "Reference node")
        real_end = subgraph.get_feature_of_node(end, "Reference node")
        to_return.add((real_start, real_end))
    return(to_return)

setattr(Graph, "smartest_network", smartest_network)


# This class finds the shortest ordered route given a sequence of nodes and based on a certain distance, if no
# distance is specified it uses the network distance
# The method uses Dijkstra when a distance is specified or Breadth-First-Search for network distance
# The function returns None if not such a route exists
def shortest_ordered_route(self, H, sequence_of_nodes, distance = None):
    route = [H]
    sequence = [H]
    sequence.extend(sequence_of_nodes)
    for index in range(len(sequence) - 1):
        if(distance is None):
            sub_route = self.breadth_first_search(sequence[index], sequence[index + 1])
        else:
            sub_route = self.dijkstra(sequence[index], sequence[index + 1], distance)
        if(sub_route is None):
            return(None)
        route.extend(sub_route[1:])
    return(route)

setattr(Graph, 'shortest_ordered_route', shortest_ordered_route)



# This function finds an approximate shortest route starting from a node H and passing through all the nodes
# in set_of_nodes, everything in a weighted graph, with weight given by weight parameter
def shortest_route_dijkstra(self, H, set_of_nodes, weight):
    if(len(set_of_nodes) == 0):
        return([])
    
    for node in set_of_nodes:
        self.add_features_to_node(node, { "In set" : True})
    heap = Binary_Heap((H, 0), lambda x : x[1])
    self.add_features_to_node(H, { "Distance" : 0 })
    while(True):
        if(heap.is_empty()):
            self.reset_features(["Visited", "Distance", "Parent", "In set"])
            return(None)
        current_node = heap.extract()
        if(self.get_feature_of_node(current_node[0], "In set")):
            break
        if(self.get_feature_of_node(current_node[0], "Visited")):
            continue
        this_distance = self.get_feature_of_node(current_node[0], "Distance")
        for close_node in self.get_neighbours(current_node[0], [weight]):
            prev_distance = self.get_feature_of_node(close_node[0], "Distance")
            arc_weight = close_node[1][weight]
            new_distance = this_distance + arc_weight
            if(prev_distance is None or prev_distance > new_distance):
                self.add_features_to_node(close_node[0], {"Distance" : new_distance, "Parent" : current_node[0]})
                heap.insert((close_node[0], new_distance))
        self.add_features_to_node(current_node[0], {"Visited" : True})
        
    current_node = current_node[0]
    set_of_nodes.remove(current_node)
    new_H = current_node
    aux_list = []
    while(not current_node is None):
        aux_list.append(current_node)
        current_node = self.get_feature_of_node(current_node, "Parent")
    self.reset_features(["Visited", "Distance", "Parent", "In set"])
    aux_list = aux_list[::-1]
    recursion_list = self.shortest_route_dijkstra(new_H, set_of_nodes, weight)
    if(recursion_list is None):
        return(None)
    aux_list.extend(recursion_list[1:])
    return(aux_list)

setattr(Graph, 'shortest_route_dijkstra', shortest_route_dijkstra)


# This function is like shortest_route_dijkstra, but with unweighted graphs, so it doesn't need a weight parameter
def shortest_route_bfs(self, H, set_of_nodes):
    if(len(set_of_nodes) == 0):
        return([])
    for node in set_of_nodes:
        self.add_features_to_node(node, {"In set" : True})
    in_queue_nodes = deque()
    in_queue_nodes.appendleft(H)
    self.add_features_to_node(H, {"Visited" : True})
    while(True):
        if(not in_queue_nodes):
            self.reset_features(["Visited", "Parent", "In set"])
            return(None)
        visiting_now = in_queue_nodes.pop()
        if(self.get_feature_of_node(visiting_now, "In set")):
            break
        for neighbour in self.get_neighbours(visiting_now):
            if(self.get_feature_of_node(neighbour, "Visited") is None):
                in_queue_nodes.appendleft(neighbour)
                self.add_features_to_node(neighbour, {"Parent" : visiting_now, "Visited" : True})
    
    set_of_nodes.remove(visiting_now)
    new_H = visiting_now
    aux_list = []
    while(not visiting_now is None):
        aux_list.append(visiting_now)
        visiting_now = self.get_feature_of_node(visiting_now, "Parent")
    aux_list = aux_list[::-1]
    self.reset_features(["Visited", "Parent", "In set"])
    recursion_list = self.shortest_route_bfs(new_H, set_of_nodes)
    if(recursion_list is None):
        return(None)
    aux_list.extend(recursion_list[1:])
    return(aux_list)

setattr(Graph, 'shortest_route_bfs', shortest_route_bfs)

# This is the function which solves our fourth functionality
def shortest_route(self, H, set_of_nodes, distance = None):
    if(distance is None):
        return(self.shortest_route_bfs(H, set_of_nodes))
    else:
        return(self.shortest_route_dijkstra(H, set_of_nodes, distance))
    
setattr(Graph, "shortest_route", shortest_route)

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"]=(15,15)
def drawGraph_Func1(result, typeDistance):
    G = nx.Graph()
    E_color=''
    
    for i in result:
        G.add_node(i)


        for i in result:

            for j in result:
                edge=CAL.get_edge(i,j,[typeDistance])
                
                if not edge==None:
                    G.add_edge(i,j)
    if not nx.is_empty(G):
        colors = ["red"] + (["cyan"] * (len(result) - 1))
        pos=nx.spring_layout(G)  #<<<<<<<<<< Initialize this only once
        nx.draw(G,pos, with_labels=True, font_size=25)
        nx.draw_networkx_nodes(G,pos,node_color= colors, node_size = 1000)
        if typeDistance=="Distance":
            nx.draw_networkx_edges(G, pos,edge_color='blue', width=3)
        elif typeDistance=="Time":
            nx.draw_networkx_edges(G, pos,edge_color='purple', width=3)
        else:
            nx.draw_networkx_edges(G, pos,edge_color='red', width=3)
        
        plt.show()
        
import matplotlib.pyplot as plt

#plt.rcParams["figure.figsize"]=(15,15)
def drawGraph_Func2(result,typeDistance,set_nodes):
    G = nx.Graph()
    for node in set_nodes:
        G.add_node(node)
    
    for i in set_nodes:

            for j in set_nodes:
                edge=CAL.get_edge(i,j,[typeDistance])
                
                if not edge==None:
                    G.add_edge(i,j)
    
    edge_colours=[]
    
    for edge in G.edges():
        edge2=(edge[1],edge[0])
        if edge in result or edge2 in result:
            if typeDistance=="Distance":
                edge_colours.append('blue')
            elif typeDistance=="Time":
                edge_colours.append('purple')
            else:
                edge_colours.append('red')
        else:
            edge_colours.append('black')
    #edge_colours = ['black' if not edge in result else 'red'
                    #for edge in G.edges()]
    
    if not nx.is_empty(G):
        pos=nx.spring_layout(G)
        
        nx.draw(G,pos, with_labels=True, font_size=20,connectionstyle='arc3,rad=0.2',edge_color=edge_colours,width=3)
        nx.draw_networkx_nodes(G,pos, node_size = 1000, node_color="cyan")
        
        
        
        
            
        
            
        plt.show()
        
        
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"]=(15,15)
def drawGraph(result):
    G = nx.DiGraph()

    for i in range(len(result)-1):
        G.add_edge(result[i],result[i+1], label=i)
        
    edge_labels = nx.get_edge_attributes(G, 'label')
    
    if not nx.is_empty(G):
        pos=nx.spring_layout(G)
        nx.draw(G,pos, with_labels=True, font_size=25,connectionstyle='arc3,rad=0.2', arrowsize=25)
        colors = ["red"] + (["cyan"] * (len(result) - 1))
        nx.draw_networkx_nodes(G,pos, node_size = 650, node_color=colors)
        
        
        
            
        plt.show()
        
        
        
        
        

func = int(input("chose your functionality (1,2,3,4): "))

if func == 1:
        
        # input
        v = int(input("insert the starting node ID: "))
        
        d = int(input('insert the threshold d: '))
        
        typeDistance= str(input('insert the type of distance(Time, Distance (for metres distance), NetworkDistance): '))
        nodeList=[]
        # running the functionality
        result_func1=getNeigh(v,nodeList,d,typeDistance)
        print("The result of functionality 1:")
        print(result_func1)
        drawGraph_Func1(result_func1,typeDistance)

elif func == 2:
         # input
       
        set_nodes = list(map(int, input("insert a set of nodes(separeted by space): ").split(" ")))
        
        typeDistance= str(input('insert the type of distance(Time, Distance (for metres distance) or empty string if you want a Netowrk distance): '))
        
        result_func2=CAL.smartest_network(set_nodes,typeDistance)
        print("The result of functionality 2:")
        print(result_func2)
        drawGraph_Func2(result_func2,typeDistance,set_nodes)

        

       
    
elif func == 3:
        # getting the input:
        node = int(input("insert the starting node ID: "))

        set_node = list(map(int, input("insert a set of nodes(separeted by space): ").split(" ")))
        
        
        
        result_func3=CAL.shortest_ordered_route(node,set_node)
        print("The result of functionality 3:")
        print(result_func3)
        drawGraph(result_func3)

elif func == 4:
        # getting the input:
        node = int(input("insert the starting node ID: "))

        set_node = set(map(int, input("insert a set of nodes(separeted by space): ").split(" ")))
        
        result_func4=CAL.shortest_route(node,set_node)
        print("The result of functionlity 4:")
        print(result_func4)
        drawGraph(result_func4)


# In[ ]:




