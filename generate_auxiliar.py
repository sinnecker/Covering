import numpy as np
import networkx as nx
import time 
from scipy.spatial import distance_matrix, Delaunay, ConvexHull, Voronoi

def sort_edge(edge):

    #return the edge with the extremes sorted
    p1, p2 = edge
    return (min(p1, p2), max(p1, p2))


def eliminate_hull(triangles, edges):
   
    #convert edges to a set of tuples with both (p1, p2) and (p2, p1) for easy lookup
    hull_set = set()
    for edge in edges:
        p1, p2 = edge
        hull_set.add(sort_edge((p1, p2)))
        

    #filter triangles
    edges_set = set()
    for triangle in triangles:
        p1, p2, p3 = triangle
        edges_set.add(sort_edge((p1, p2)))
        edges_set.add(sort_edge((p2, p3)))
        edges_set.add(sort_edge((p3, p1)))

    filtered_edges = edges_set - hull_set
    

    return list(filtered_edges)

def check_intersection(edge,circle):
    
    p1,p2 = edge
    R,center = circle
    
    #edge vector
    vector = p2-p1
    
    #vector from the extreme of the edge to the center of the circle
    u = p1-center
    
    #quadratic coefficients
    a = np.dot(vector,vector)
    b = 2*np.dot(vector,u)
    c = np.dot(u,u) - R**2

    #discriminant
    delta = (b**2 - 4*a*c)
    
    #no intersection
    if delta<0:
        return 0,None

    # 2 intersections and computes the intersection
    return 2, ((-b-np.sqrt(delta))/(2*a),(-b+np.sqrt(delta))/(2*a))


def generate_data(edges,circles):
    
    #initialize empty matrix
    A = np.zeros((0, len(circles)))  
    New_edges = []

    #loop on the graph edges
    for edge in edges:

        p1,p2 = edge
        #initialize the subintervals
        subintervals = [0,1]
        row = []
        
        #loop on the circles to find all intersections
        for i,circle in enumerate(circles):
            #search for intersection
            n,intersections = check_intersection([edge[0],edge[1]],circle)
            if n==2:
                #check if the projection is in [0,1]
                for i in intersections:
                    if 0<= i <= 1:
                        if i not in subintervals:
                            subintervals.append(i)

        #generates all intervals representaing each intersection 
        subintervals.sort()
        #computes the new edges and find the rows of Matrix A
        for start, end in zip(subintervals[:-1], subintervals[1:]):
            new_edge_start = p1 + start * (p2 - p1)
            new_edge_end = p1 + end * (p2 - p1)
            new_edge = np.array([new_edge_start,new_edge_end])

            row = []
            for circle in circles:
                check,_ = check_intersection(new_edge,circle)
                row.append(1 if check else 0)

            A = np.vstack([A, row])

            New_edges.append(new_edge)
    return A,New_edges

def generate_problem(nodes, circles, rmin, rmax, method=1):
    
    #generate random nodes
    Vs = np.random.rand(nodes,2)
    
    #generate random circles center and radii
    Ps = np.random.rand(circles,2)
    Rs = np.random.uniform(rmin,rmax,circles)

    #generate weights
    Ws = np.random.uniform(0.5,1.5)*Rs**2
    
    #compute the distance matrix of the nodes
    D = distance_matrix(Vs,Vs)
    
    #computes the graph, MST, delaunay triangulation, convex hull
    G1 = nx.from_numpy_array(D)
    T1 = nx.minimum_spanning_tree(G1)
    delauny_triag = Delaunay(Vs)
    hull = ConvexHull(Vs)

    #get rid off hull edges
    delauny = eliminate_hull(delauny_triag.simplices,hull.simplices) 

    #generates circle data
    circles_data = [(Rs[i],Ps[i]) for i in range(p)]

    #generate edges
    edges = list(T1.edges)+delauny
    Nedges = len(edges)
    
   
    A, New_edges = generate_data(edges,circles_data)
    NP = 1
    #while the problem is infeasible
    while np.any(np.sum(A, axis=1)<1):
        NP += 1
        
        if method==1:
            circles = int(np.round(circles*1.1))

        if method==2:
            R_min = R_min*1.1
            R_max = R_max*1.1
        
        Ps = np.random.rand(circles,2)
        Rs = np.random.uniform(rmin,rmax,circles_data)

        Ws = np.random.uniform(0.5,1.5)*Rs**2
        circles = [(Rs[i],Ps[i]) for i in range(p)]
        A, New_edges = generate_data(edges,circles_data)

    print("Number of problems generated: ",NP)")

    return A, edges, New_edges, circles_data, Ws, circles, Vs, rmin, rmax, Nedges



