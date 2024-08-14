import numpy as np
from gurobipy import*


def Generate_CP_model(A,circles,Ws):
    
    M = Model()
    x = M.addMVar(shape=circles, vtype=GRB.BINARY, name="x")
    M.addConstr(A @ x >= 1, name="c")
    M.setObjective(Ws@x, GRB.MINIMIZE)

    return M,x

def Generate_CP2_model(Ws,circle_data,edges_data,square_size):
    
    n = len(circle_data)
    E = len(edges_data)
    
    M = Model()
    
    x = M.addMVar(shape=(n,2), vtype=GRB.CONTINUOUS, name="x")
    y = M.addMVar(shape=(n,E), vtype=GRB.BINARY, name="y")
    z = M.addMVar(shape=n, vtype=GRB.BINARY, name="z")
    
    for j in range(n):
        M.addConstrs(z[j] >= y[j][i] for i in range(E))
    
    for j in range(E):
        M.addConstr(sum(y[i][j] for i in range(n))>=1)
    
    for i in range(n):
        for j in range(E):
            M.addConstr((x[i]-edges_data[j][0])@(x[i]-edges_data[j][0])<= circle_data[i][0]**2 + (2 - circle_data[i][0]**2)*(1-y[i][j]))
            M.addConstr((x[i]-edges_data[j][1])@(x[i]-edges_data[j][1])<= circle_data[i][0]**2 + (2 - circle_data[i][0]**2)*(1-y[i][j]))
    M.addConstrs(x[i][0]<=  circle_data[i][1][0]+square_size/2 for i in range(n))
    M.addConstrs(x[i][1]<=  circle_data[i][1][1]+square_size/2 for i in range(n))
    M.addConstrs(x[i][0]>=  circle_data[i][1][0]-square_size/2 for i in range(n))
    M.addConstrs(x[i][1]>=  circle_data[i][1][1]-square_size/2 for i in range(n))
    
    M.setObjective(Ws@z, GRB.MINIMIZE)
    
    return M,x,y,z