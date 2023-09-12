"""
====================================================+
Assignment 2                                        |
----------------------------------------------------| 
Name:       : Garret Yong Shern Min                 |
Student ID  : 31862616                              |
Unit Code   : FIT2004                               |
====================================================+
"""

"""
This marks the start of resources section (data structures to aid assignment, not part of assignment)
=====|Resources Start|===============================================================================
"""

"""
This immediate following section is a copy of Referential_Array copied from FIT2085
"""
__author__ = "Julian Garcia for the __init__ code, Maria Garcia de la Banda for the rest"
__docformat__ = 'reStructuredText'

from ctypes import py_object
from graphlib import TopologicalSorter
from typing import TypeVar, Generic

T = TypeVar('T')


class ArrayR(Generic[T]):
    def __init__(self, length: int) -> None:
        """ Creates an array of references to objects of the given length
        :complexity: O(length) for best/worst case to initialise to None
        :pre: length > 0
        """
        if length <= 0:
            raise ValueError("Array length should be larger than 0.")
        self.array = (length * py_object)() # initialises the space
        self.array[:] =  [None for _ in range(length)]

    def __len__(self) -> int:
        """ Returns the length of the array
        :complexity: O(1)
        """
        return len(self.array)

    def __getitem__(self, index: int) -> T:
        """ Returns the object in position index.
        :complexity: O(1)
        :pre: index in between 0 and length - self.array[] checks it
        """
        return self.array[index]

    def __setitem__(self, index: int, value: T) -> None:
        """ Sets the object in position index to value
        :complexity: O(1)
        :pre: index in between 0 and length - self.array[] checks it
        """
        self.array[index] = value


"""
The following section is a modified version of Max_Heap copied from FIT1008 - 2085

This class has been since been modified into a Min_Heap and modified further to meet the requirements of the specification
"""
class Heap(Generic[T]):
    MIN_CAPACITY = 1

    def __init__(self, max_size: int) -> None:
        self.length = 0
        self.the_array = ArrayR(max(self.MIN_CAPACITY, max_size) + 1)
        self.index_array = ArrayR(max(self.MIN_CAPACITY, max_size) + 1)

    def __len__(self) -> int:
        return self.length

    def is_full(self) -> bool:
        return self.length + 1 == len(self.the_array)

    def rise(self, k: int) -> None:
        """
        Rise element at index k to its correct position
        :pre: 1 <= k <= self.length
        """

        item = self.the_array[k]
        index = self.index_array[k]
        if self.the_array[k // 2] != None:
            hold = self.index_array[self.the_array[k // 2][0]+1]
        change_to = index 
        
        while k > 1 and item[1] < self.the_array[k // 2][1]:
            
            self.index_array[self.the_array[k // 2][0]+1] = change_to

            self.the_array[k] = self.the_array[k // 2]           
            k = k // 2
            change_to = hold
            if self.the_array[k // 2] != None:
                hold = self.index_array[self.the_array[k // 2][0]+1]
        self.the_array[k] = item
        self.index_array[index] = k


    def add(self, element: T) -> bool:
        """
        Swaps elements while rising
        """
        if self.is_full():
            raise IndexError

        self.length += 1
        self.the_array[self.length] = element
        self.index_array[self.length] = self.length
        self.rise(self.length)

    def get_min(self):
        """
        Removes the maximum element from the heap, returning it.
        :pre: Heap is non-empty.
        """
        if self.length == 0:
            raise IndexError

        min_elt = self.the_array[1]
        min_index = min_elt[0]+1
        self.length -= 1

        if self.length > 0:
            self.index_array[self.the_array[self.length+1][0]+1] = 1
            self.the_array[1] = self.the_array[self.length+1]
            self.sink(1)

        self.the_array[self.length+1] = min_elt
        self.index_array[min_index] = self.length+1
        
        return min_elt

    def largest_child(self, k: int) -> int:
        """
        Returns the index of k's child with greatest value.
        :pre: 1 <= k <= self.length // 2
        """
        # TODO
        if self.length == 2*k or self.the_array[2*k][1] < self.the_array[2*k+1][1]:
            return 2*k
        else:
            return 2*k+1

    def sink(self, k: int) -> None:
        """ Make the element at index k sink to the correct position """
        # TODO

        item = self.the_array[k]
        index = self.index_array[item[0] +1]
        while 2*k <= self.length and item[1] > self.the_array[self.largest_child(k)][1]:
            self.the_array[k] = self.the_array[self.largest_child(k)]
            self.index_array[self.the_array[self.largest_child(k)][0]+1] = index
            k = self.largest_child(k)

        self.the_array[k] = item 
        self.index_array[item[0]+1] = k

"""
=====|Resources End|=================================================================================
"""




"""
This marks the start of Task 1 of Assignment 2
=====|Task 1 Start|==================================================================================
"""
"""
This class is the class used to represent edges between two vertices 
"""
class Road:
    def __init__(self,u ,v, w: int):
        # start
        self.u = u
        # end
        self.v = v
        # weight
        self.w = w

    def __str__(self):
        output_str = str(self.u) + " ," + str(self.v) + " ," + str(self.w)
        return output_str

"""
This class is used to represent a vertex of a graph
"""
class Location:
    def __init__(self, id):
        self.id = id
        self.edges = []  # list of edges
        self.reverse_edges = [] # list of reverse edges 

        # Statuses for traversal
        self.discovered = False
        self.visited = False

        # Distance of vertices from source node
        self.distance = 0

        # Backtrack (previous vertex)
        self.previous = None

        # Wait time of the location
        self.waiting_time = 0

        # Boolean indeicator for cafe/ no cafe status
        self.is_cafe = False

    def add_edge(self, edge_to_add: Road):
        self.edges.append(edge_to_add)
    
    def add_reverse_edge(self, edge_to_add: Road):
        self.reverse_edges.append(edge_to_add)

    def add_cafe(self, waiting_time):
        self.is_cafe = True
        self.waiting_time = waiting_time

    def discovered_vertex(self):
        self.discovered = True
    def reset_discovered(self):
        self.discovered = False

    def visited_vertex(self):
        self.visited = True
    def reset_visited(self):
        self.visited = False

    def reset_discovered_visited(self):
        self.reset_discovered()
        self.reset_visited()

    def __str__(self):
        output_str = str(self.id)
        for edge in self.edges:
            output_str = output_str + "\n with edge(s):" + str(edge)
        return output_str


import math

"""
This class is used to represent a graph where the vertices are locations (which then could be cafes) then the edges represent roads that link each location/ cafe
"""
class RoadGraph:
    """
    This is the constructor for the RoadGraph class

    Time complexity         : O(|V | + |E|)
    Aux Space complexity    : O(E)
    """
    def __init__(self, roads, cafes):
        # Finding the max index     O(E)
        max_index = 0
        for i in range(0,2):
            for location in roads:
                if location[i] > max_index:
                    max_index = location[i]

        # Setting up the number of locations (empty)
        self.vertices = [None] * (max_index+1)

        # Setting up locations      O(V)
        for i in range(max_index+1):
            self.vertices[i] = Location(i)

        # Determining and allocating cafes
        self.cafe_locations = []
        for cafe in cafes:
            self.vertices[cafe[0]].add_cafe(cafe[1])
            self.cafe_locations.append(cafe[0])


        self.add_edges(roads)
        self.add_reverse_edges(roads)

    def __str__(self):
        output_str = ""
        for vertex in self.vertices:
            output_str = output_str + "Vertex " + str(vertex) + "\n"
        return output_str

    """
    This method takes in a list of roads and allocates it based on their start location, u

    Time complexity         : O(E)
    Aux Space complexity    : O(E)
    """
    def add_edges(self, edges_list):
        for edge in edges_list:
            u = edge[0]
            v = edge[1]
            w = edge[2]

            current_edge = Road(u, v, w)
            current_vertex = self.vertices[u]
            current_vertex.add_edge(current_edge)
    
    """
    This method takes in a list of roads and allocates it based on their end location, v

    Time complexity         : O(E)
    Aux Space complexity    : O(E)
    """
    def add_reverse_edges(self, edges_list):
        for edge in edges_list:
            u = edge[0]
            v = edge[1]
            w = edge[2]

            current_edge = Road(v, u, w)
            current_vertex = self.vertices[v]
            current_vertex.add_reverse_edge(current_edge)

    """
    This is a modified dijkstra algorithm that is used to find the distance between all other points and the source point

    Time complexity         : O(E logV)
    Aux Space complexity    : O(E + V)
    """
    def dijkstra(self, source, destination):
        """
        Function of Dijkstra
        """
        # Instantiate a min heap to store all vertices
        discovered_nodes = Heap(len(self.vertices)+1)
        
        # Go through all vertices, reset all data then add to min heap
        for vertex in self.vertices:
            vertex.reset_discovered()
            vertex.reset_visited()
            vertex.previous = None
            vertex.distance = math.inf
            discovered_nodes.add((vertex.id,vertex.distance))
        
        # Allocating the source node (beginning point) data
        source_node = self.vertices[source]
        source_distance = 0
        source_node.distance = source_distance
        source_node.visited_vertex()
        source_node.discovered_vertex()

        # Insert the source node into its rightful place then update the heap
        discovered_nodes.the_array[discovered_nodes.index_array[source_node.id +1 ]] = (source_node.id, source_node.distance)     # updates the new min distance 
        discovered_nodes.rise(discovered_nodes.index_array[source_node.id+1])  

        # While not all nodes have been visited, iterate to get shortest distance
        while len(discovered_nodes) > 0:
            # Serve smallest item
            current_node = self.vertices[discovered_nodes.get_min()[0]] 
            current_node.visited_vertex()                                           #set .visited = True
            
            # No use
            if current_node == destination:
                return 

            # Iterate through every edge available 
            for edge in current_node.edges:
                v = self.vertices[edge.v]

                # If not discovered yet then allocate distance and previous
                if v.discovered == False:
                    v.discovered_vertex()                           #set .discovered = True
                    v.distance = current_node.distance + edge.w
                    v.previous = current_node                     
                    discovered_nodes.the_array[discovered_nodes.index_array[v.id +1 ]] = (v.id, v.distance)     # updates the new min distance 
                    discovered_nodes.sink(discovered_nodes.index_array[v.id+1])
                    discovered_nodes.rise(discovered_nodes.index_array[v.id+1])                          # Since new update will always be smaller, make element rise/climb
                    
                # In heap but not finalised, update distance and previous
                elif v.visited == False:
                    if v.distance > current_node.distance + edge.w:
                        # update distance
                        v.distance = current_node.distance + edge.w
                        v.previous = current_node  
                        #update heap 
                        discovered_nodes.the_array[discovered_nodes.index_array[v.id +1 ]] = (v.id, v.distance)     # updates the new min distance 
                        discovered_nodes.sink(discovered_nodes.index_array[v.id+1])
                        discovered_nodes.rise(discovered_nodes.index_array[v.id+1])                        # Since new update will always be smaller, make element rise/climb


    """
    Similar to the dijsktra above but instead of analysing from source to the destination; this is from destination to source
    
    Time complexity         : O(E logV)
    Aux Space complexity    : O(E + V)
    """
    def reversed_dijkstra(self, source, destination):
        """
        Function of Dijkstra (but reversed, ooh)
        """
        # Instantiate a min heap to store all vertices
        discovered_nodes = Heap(len(self.vertices)+1)
        
        # Go through all vertices, reset all data then add to min heap
        for vertex in self.vertices:
            vertex.reset_discovered()
            vertex.reset_visited()
            vertex.distance = math.inf
            vertex.previous = None
            discovered_nodes.add((vertex.id,vertex.distance))
        
        # Allocating the source node (beginning point) data
        source_node = self.vertices[source]
        source_distance = 0
        source_node.distance = source_distance
        source_node.visited_vertex()
        source_node.discovered_vertex()

        # Insert the source node into its rightful place then update the heap
        discovered_nodes.the_array[discovered_nodes.index_array[source_node.id +1 ]] = (source_node.id, source_node.distance)     # updates the new min distance        
        discovered_nodes.rise(discovered_nodes.index_array[source_node.id+1])    
        
        # While not all nodes have been visited, iterate to get shortest distance
        while len(discovered_nodes) > 0:
            # Serve smallest item
            current_node = self.vertices[discovered_nodes.get_min()[0]] 
            current_node.visited_vertex()                                           #set .visited = True
            
            # No use
            if current_node == destination:
                return 

            # Iterate through every edge available 
            for edge in current_node.reverse_edges:
                v = self.vertices[edge.v]
                
                # If not discovered yet then allocate distance and previous
                if v.discovered == False:
                    v.discovered_vertex()                           #set .discovered = True
                    v.distance = current_node.distance + edge.w
                    v.previous = current_node
                    discovered_nodes.the_array[discovered_nodes.index_array[v.id +1 ]] = (v.id, v.distance)     # updates the new min distance 
                    discovered_nodes.sink(discovered_nodes.index_array[v.id+1])
                    discovered_nodes.rise(discovered_nodes.index_array[v.id+1])                           # Since new update will always be smaller, make element rise/climb


                # In heap but not finalised, update distance and previous
                elif v.visited == False:

                    if v.distance > current_node.distance + edge.w:
                        # update distance
                        v.distance = current_node.distance + edge.w
                        v.previous = current_node  
                        #update heap 
                        discovered_nodes.the_array[discovered_nodes.index_array[v.id +1 ]] = (v.id, v.distance)     # updates the new min distance 
                        discovered_nodes.sink(discovered_nodes.index_array[v.id+1])
                        discovered_nodes.rise(discovered_nodes.index_array[v.id+1])                        # Since new update will always be smaller, make element rise/climb
                

    """
    This method is used to find the shortest distance between the start and end points
    ===============================================================================================
    This method involves using 2 dijkstra algorithms to find the shortest distance between start
    and end while going through at least 1 cafe while only stopping to get coffee at one

    Time complexity         : O(|E| log|V|)
    Aux Space complexity    : O(|V | + |E|)
    """
    def routing(self, start, end):
        
        # Instantiating two lists for the pathing taken to get coffee and to destination
        # Together this produces an Aux Space Complexity: O(E)
        valid_routes_to_cafe = []
        valid_routes_from_cafe = []

        # Conduct a dijkstra to the destination
        self.dijkstra(start, end)
        
        # Note the pathing to all cafes
        # NOTE: The pathing format is [cafe id, distance, pathing, pathing, pathing, ...]
        for cafe_location in self.cafe_locations:
            # Instantiating output list
            temp_list = []
            # Instantiating a list to store info that will need to be reversed later
            temp_list_reverse = []
            
            start_backtrack = self.vertices[cafe_location]
            while start_backtrack != None:
                temp_list_reverse.append(start_backtrack.id) 
                start_backtrack = start_backtrack.previous

            # Append the distance then the cafe id 
            temp_list_reverse.append( (self.vertices[cafe_location].distance + self.vertices[cafe_location].waiting_time) )
            temp_list_reverse.append(cafe_location)

            # Reverese the pathing
            temp_list = temp_list_reverse[::-1]

            # Append the route that leads TO the cafe
            valid_routes_to_cafe.append(temp_list)
        

        # Conduct a dijkstra to the start
        self.reversed_dijkstra(end, start)

        # Note the pathing to all cafes
        for cafe_location in self.cafe_locations:
            # Instantiating output list
            temp_list = []
            # No reverse list because the info can be appended in the right order

            # Append the cafe id then distance
            temp_list.append(cafe_location)
            temp_list.append( (self.vertices[cafe_location].distance) )
            
            start_backtrack = self.vertices[cafe_location]
            while start_backtrack != None:
                temp_list.append(start_backtrack.id)
                start_backtrack = start_backtrack.previous
            
            # Append the route that leads FROM the cafe
            valid_routes_from_cafe.append(temp_list)

        # Instantiate the shortest distance for comparison 
        final_distance = math.inf
        # Instantiate the list to store the shortest path
        final_path = []
        # Instantiate the optimal cafe to visit for shortest path
        optimal_cafe = -1

        # Iterate through the cafe locations and sum the distances and compare it to find the optimal cafe to visit for shortest path 
        for i in range(len(self.cafe_locations)):
            if valid_routes_to_cafe[i][0] == valid_routes_from_cafe[i][0]:
                if final_distance > valid_routes_to_cafe[i][1] + valid_routes_from_cafe[i][1]:
                    final_distance = valid_routes_to_cafe[i][1] + valid_routes_from_cafe[i][1]
                    optimal_cafe = i

        # Based on the optimal cafe to visit for shortest path, merge the pathing TO and FROM to get optimal pathing
        for j in range(2,len(valid_routes_to_cafe[optimal_cafe])):
            final_path.append(valid_routes_to_cafe[optimal_cafe][j])
        # Start at 3rd element to exclude cafe as it was included in the above loop
        for j in range(3,len(valid_routes_from_cafe[optimal_cafe])):
            if final_path[-1] != valid_routes_from_cafe[optimal_cafe][j]:
                final_path.append(valid_routes_from_cafe[optimal_cafe][j])
        
        # Output the optimal pathing 
        return final_path
"""
=====|Task 1 End|====================================================================================
"""


"""
This marks the start of Task 2 of Assignment 2
=====|Task 2 Start|==================================================================================
"""
"""
This class is used to construct a graph that will be used to manage all the intersection points (vertices) and downhill segments (edges)
"""
class SkiGraph:
    def __init__(self, downhillScores):
        # Instantiate a min heap to store all vertices
        max_index = 0
        for i in range(0,2):
            for location in downhillScores:
                if location[i] > max_index:
                    max_index = location[i]

        # Setting up the number of locations (empty)
        self.vertices = [None] * (max_index+1)

        # Setting up locations      O(V)
        for i in range(max_index+1):
            self.vertices[i] = Location(i)

        self.add_edges(downhillScores)

    def __str__(self):
        output_str = ""
        for vertex in self.vertices:
            output_str = output_str + "Vertex " + str(vertex) + "\n"
        return output_str

    """
    This method takes in a list of roads and allocates it based on their start location, u

    Time complexity         : O(E)
    Aux Space complexity    : O(E)
    """
    def add_edges(self, edges_list):
        for edge in edges_list:
            u = edge[0]
            v = edge[1]
            w = edge[2]

            current_edge = Road(u, v, w)
            current_vertex = self.vertices[u]
            current_vertex.add_edge(current_edge)

    """
    This method is a topological sort helper to allow for recursion to achieve the sort 

    Time complexity         : O(D)
    Aux Space complexity    : O(1)
    """
    def topological_sort_helper(self, current_vertex, output_stack):
        current_vertex.visited_vertex()

        for edge in current_vertex.edges:
            if self.vertices[edge.v].visited == False:
                self.topological_sort_helper(self.vertices[edge.v], output_stack)

        output_stack.append(current_vertex)
    
    """
    The method is a topological sort to find the topological order to visit each intersection points

    Time complexity         : O(D * P)
    Aux Space complexity    : O(P)
    """
    def topological_sort(self):
        # Instantiating the output list for the sort
        output_stack = []

        # Iterate though all the points 
        for vertex in self.vertices:
            if vertex.visited == False:
                self.topological_sort_helper(vertex,output_stack)
        
        output_list = output_stack[::-1]
        return output_list

"""
This method is the method to find the optimal skiing route to score the maximum points using Dynamic Programming practices

Time complexity         : O(D * P)
Aux Space complexity    : O(P)
"""
def optimalRoute(downhillScores, start, finish):
        # Instantiating a graph for computation
        ski_graph = SkiGraph(downhillScores)

        # Instantiating a memo list to reduce recomputation
        memo = [-1 * math.inf] * len(ski_graph.vertices)

        # Find the topological order of the graph (all the intersection points)
        vertice_order = ski_graph.topological_sort()

        # marking distance of start point as 0
        memo[start] = 0 

        # Iterate throught all intersection points 
        for vertex in vertice_order:
            u = vertex
            # Iterate through all downhill segments and update distances in the memo
            for edge in u.edges:
                v = ski_graph.vertices[edge.v]
                score = edge.w + memo[u.id]

                if score > memo[v.id]:
                    memo[v.id] = score
                    v.previous = u
        
        # Instantiate a list to store the optimal path
        longest_path = []
        # Initiate variable to start backtracking 
        start_backtrack = ski_graph.vertices[finish]
        # Start backtracking
        while start_backtrack != None:
            longest_path.append(start_backtrack.id)
            start_backtrack = start_backtrack.previous
        
        # Reverse the path to find the forward path (start -> finish)
        final_output = longest_path[::-1]
        # Output the best path
        return final_output