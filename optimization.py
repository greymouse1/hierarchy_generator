from gurobipy import *
import networkx as nx
from tqdm import tqdm
import pickle


# Load graph from file
weighted_graph = nx.read_graphml("jaccard_index.graphml")

# Load trees
with open("T1.pkl", "rb") as f:
    T1 = pickle.load(f)
with open("T2.pkl", "rb") as f:
    T2 = pickle.load(f)

# Run Gurobi optimization

# Get the set of all edges
all_edges = list(weighted_graph.edges())
# Create a dictionary to map edges to their indexes
edge_to_index = {edge: i for i, edge in enumerate(all_edges)}

# Create the model
optimization_model = Model("bipartite solver")

# Add decision variables x
x = optimization_model.addVars(len(all_edges), vtype=GRB.BINARY, name="x")

# Define objective function
obj_function = LinExpr([weighted_graph.edges[edge]['weight'] for edge in all_edges], x.values())
optimization_model.setObjective(obj_function, GRB.MAXIMIZE)

# Set constraints for T1
for path in tqdm(T1.leaf_list, desc=f"Setting constraints for {T1}"):
    indexes = [edge_to_index[edge] for edge in weighted_graph.edges(path)]
    constraint_sum = LinExpr([1] * len(indexes), [x[i] for i in indexes])
    optimization_model.addConstr(constraint_sum <= 1)

# Set constraints for T2
for path in tqdm(T2.leaf_list, desc=f"Setting constraints for {T2}"):
    indexes = [edge_to_index[edge[::-1]] for edge in weighted_graph.edges(path)]
    constraint_sum = LinExpr([1] * len(indexes), [x[i] for i in indexes])
    optimization_model.addConstr(constraint_sum <= 1)

# Optimize
optimization_model.optimize()

matched_indexes = []
for v in optimization_model.getVars():
    if v.x == 1:
        #print( '%s: %g ' % (v.varName,v.x))
        matched_indexes.append(int(v.varName.split('[')[1].split(']')[0]))
matched_edges = []
for i in matched_indexes:
    edge = all_edges[i]
    matched_edges.append(edge)

# Define function for calculating number of leafs
def number_of_leaf_nodes(tree, node):
    descendants = nx.descendants(tree, node)
    leaf_nodes = [n for n in descendants if tree.out_degree(n) == 0]
    # this is because amount of leaves will be either 0 which means the match is
    # one of the base polygons, or 2 or more. It can't be 1 because parent node
    # is by default made by merging two or more polygons
    if len(leaf_nodes) >= 2:
        return len(leaf_nodes)
    else:
        return 1

for first,second in matched_edges:
    m = number_of_leaf_nodes(T1.G,first)
    n = number_of_leaf_nodes(T2.G,second)
    print(f"Edge {first}:{second} is a {m}:{n} match")




