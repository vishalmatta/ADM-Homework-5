#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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



# This is the function which solves our fourth functionality
def shortest_route(self, H, set_of_nodes, distance = None):
    if(distance is None):
        return(self.shortest_route_bfs(H, set_of_nodes))
    else:
        return(self.shortest_route_dijkstra(H, set_of_nodes, distance))
    
setattr(Graph, "shortest_route", shortest_route)

