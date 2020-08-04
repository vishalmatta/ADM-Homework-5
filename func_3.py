#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

