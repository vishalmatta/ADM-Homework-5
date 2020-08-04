#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

