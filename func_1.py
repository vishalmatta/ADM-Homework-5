#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

