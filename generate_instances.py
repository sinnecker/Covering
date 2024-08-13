import argparse
import numpy as np
import os
from scipy.io import savemat
from generate_auxiliar import generate_problem

def main(nodes, circles, rmin, rmax, path, method=1):

    A, edges, New_edges, circles_data, weights, circles, nodes_data, rmin, rmax, Nedges = generate_problem(nodes, circles, rmin, rmax, method)

    assert(np.all(np.array([sum(k) for k in A.T])>0))

    Save_data = {}
    
    Save_data["N_nodes"]=nodes
    Save_data["R_max"] = rmax
    Save_data["R_min"] = rmin
    Save_data["Nodes"] = nodes_data
    Save_data["weights"] = weights
    Save_data["Matrix"] = A
    Save_data["Circles"] = circles_data
    Save_data["N_circles"] = circles
    Save_data["N_edges"] = Nedges
    Save_data["N_nodes"] = nodes
    Save_data["N_constraints"] = A.shape[0]
    Save_data["one_rows"] = len(np.where(np.array([sum(k) for k in A])==1)[0])
    Save_data["Full_Edges"] = edges

    savemat(path, Save_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Set Covering Problem Instance")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes")
    parser.add_argument("--circles", type=int, required=True, help="Number of circles")
    parser.add_argument("--rmin", type=float, required=True, help="Minimum radius of circles")
    parser.add_argument("--rmax", type=float, required=True, help="Maximum radius of circles")
    parser.add_argument("--path", type=str, required=True, help="Path to save the output data folder ")
    parser.add_argument("--method", type=int, default=1, help="Method for instance generation (default: 1)")
    
    args = parser.parse_args()

    main(args.nodes, args.circles, args.rmin, args.rmax, args.path, args.method)