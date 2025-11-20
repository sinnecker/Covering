import numpy as np
from gurobipy import*



def check_dominance(r1,r2,dom):
    #Checks if row r1 dominates r2
    greater = False
    for a, b in zip(r1, r2):
        if a < b:
            return False
        if a > b :
            greater = True
    if not greater:
        if np.all(r1-r2==0) and dom:
            greater = True
    return greater

def Row_reduction(A):
    #eliminates dominat rows
    sorted_matrix = A[np.argsort(A.sum(axis=1))]
    ones = []
    map = np.argsort(A.sum(axis=1))
    dominant = [True] * len(A)
    non_dom = []
    
    for i,r1 in zip(map,sorted_matrix):
        if sum(r1)==1:
            ones.append(i)
            continue
        
        for j in non_dom:
            r2 = A[j]
            if check_dominance(r1,r2,dominant[j]):
                dominant[i] = False
                
            if check_dominance(r2,r1,dominant[i]):
                dominant[j] = False
                
        if dominant[i]:
            non_dom.append(i)
        non_dom = [k for k in non_dom if dominant[k]]
        
    return dominant,ones

def Col_elim(A,ones,dominant):
    #eliminates columns
    Fix = []
    for k in ones:
        var = int(np.where(A[k]==1)[0])
        if var not in Fix:
            Fix.append(var)
            dominant[k] = False
    
    return dominant, Fix

def Reduction_simple(A):
    #simple matrix reduction
    dominant,ones = Row_reduction(A)
    dominant,Fix = Col_elim(A,ones,dominant)

    return dominant,Fix


def create_model(A,w,log=True):
    #creates the stong fixing model

    M_RC = Model()
    #M_RC.Params.CoverCuts = 2
    #M_RC.Params.Threads = 8
    #M_RC.Params.Method = 0
    if not log:
        M_RC.Params.LogToConsole = 0
    u_RC = M_RC.addMVar(shape= A.shape[0], lb = 0,  name="u")
    M_RC.addConstr(u_RC@A <= w, name="c")
    
    return M_RC,u_RC



def order_function(Columns,C):
    #jaccard distance
    p1 = np.dot(Columns,C)
    p2 = np.sum(Columns,axis=1)+np.sum(C) - p1
    return np.argsort(1-p1/p2)

def Strong_fixing(A,ws,circles,UB,order_function):
    
    #creates the model for the strong fixing procedure
    M_RC,u_RC = create_model(A,ws,False)
    
    #gets all the centers of the possible SLSs candidates
    Centers = np.concatenate(circles[:,1]).reshape(len(circles),2)
    
    #list to store the fixed variables
    Fixed = []

    #mapping the order of the variables (the order will change depending on the column chosen as the first problem)
    
    Columns = A.T
    variable = Columns[0]#sett the first column as the first problem
    mapp = np.arange(0,len(Columns))
    Order = order_function(Columns,variable)
    Columns = Columns[Order]
    mapp = mapp[Order]

    for i in range(len(Centers)):
        i = mapp[0]
        variable = Columns[0]
        if i not in Fixed:
            #updates the objective value
            M_RC.setObjective(sum(u_RC*(1-newA[:,i])),GRB.MAXIMIZE)
            M_RC.optimize()
            #numerical error treatment
            solution = np.array([0 if np.isclose(k,0) or k<0 else k for k in u_RC.X])
            #computes the reduced costs
            rc = newWs - newA.T@solution
            #store all fixed variables
            for k in np.where(rc > Data["OV"] - Ffix - sum(solution) + 1e-6)[0]:
                if k not in Fixed:
                    Fixed.append(k)
        
        if len(mapp)>1:
            
            Columns = Columns[1:]
            mapp = mapp[1:]
            Order = order_function(Columns,variable)
            Columns = Columns[Order]
            mapp = mapp[Order]

    #do the matrix reduction after strong fixing
    Colcutt = [k for k in range(newA.shape[1]) if k not in Fixed]
    newWs =  ws[Colcutt]
    newA = A[: , Colcutt]
           
    newA,Colcutt,Fix = Reduction_simple(newA,newWs)
    Ffix = sum(newWs[Fix])#possible fixed 1 variables
    newWs =  newWs[Colcutt]

    return newA,newWs,Fixed