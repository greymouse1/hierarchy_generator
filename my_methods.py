# Import the Dataset class from dataset.py
from dataset import Dataset
from tree_generator import treeGenerator, jaccardIndex
from gurobipy import *
import networkx as nx

# Instantiate the Dataset class
dataset1 = Dataset(name='zentrum_atkis',path='/Users/shark/Desktop/My Documents/uni/Munster/Possible_thesis/Bonn/virtual_folder/pythonProject/tri/zentrum_atkis',epsilon=0)
dataset2 = Dataset(name='zentrum_osm',path='/Users/shark/Desktop/My Documents/uni/Munster/Possible_thesis/Bonn/virtual_folder/pythonProject/tri/zentrum_osm',epsilon=0)

# Load data with eps=0
dataset1.loadData(0)
dataset2.loadData(0)
# Get wkt in file
dataset1.get_wkt()
dataset2.get_wkt()

# Get wkt unions in file - not really necessary anymore
# dataset.get_wkt_unions()

# Generate tree
# Right now I can use "dataset" class instance from above, I could as well
# use .txt file directly which was created with dataset.get_wkt()
# insert tree name after file name
T1 = treeGenerator(dataset1.all_wkt,"T1")
T2 = treeGenerator(dataset2.all_wkt,"T2")

# Calculate weights
# Weights will be calculated between two trees, T1 and T2
# and each weight is associated with an edge e which connects vertex
# vi from tree T1 and vj from tree T2
# so first I would run treeGenerator twice, once for each tree
# T1 = treeGenerator(wkt_t1)
# T2 = treeGenerator(wkt_t2)
# where wkt are files with triangles created with .get_wkt() method or pulling
# directly from dataset object (two dataset objects, one for each tree
# Then I would call jaccartIndex() method which would take in both trees
# and return weights for edges
# jaccartIndex() needs name of node from T1, name of node from T2 as that will
# be nomenclature used for the edge, and coordinates of polygons associated with
# each vertex which means the graph returned by treeGenerator has to have coordinates
# Graph of each tree contains edges between vertices and vertices. This edges should be
# ignored and only vertices taken into account when jaccartIndex is performed

weighted_graph = jaccardIndex(T1,T2)
print(weighted_graph)

nx.write_weighted_edgelist(weighted_graph, "graph_data.csv")


# Run Gurobi optimization

# Set the data
# First list of weights
w = [weighted_graph.edges[edge]['weight'] for edge in weighted_graph.edges()]
# Second set of edges, which have aligned indexes with weights
e = [edge for edge in weighted_graph.edges()]
# Set number of all edges
N = len(w)
# Create the model
optimization_model = Model("bipartite solver")
# Add decision variables x
x = optimization_model.addVars(N,vtype=GRB.BINARY, name="x")
# Define objective function
obj_function = sum(w[i]*x[i] for i in range(N))
optimization_model.setObjective(obj_function,GRB.MAXIMIZE)
# Set constraints
def setConstraints(input_tree):
    for path in input_tree.leaf_list:
        edges = weighted_graph.edges(path)
        indexes = [e.index(tuple_) for tuple_ in edges if tuple_ in e]
        constraint_sum = sum(x[i] for i in indexes)
        optimization_model.addConstr(constraint_sum <= 1)
        leaf_node = path[-1]
        print(f"Constraint sum for path {path}: {constraint_sum}")

for tree in [T1,T2]:
    setConstraints(tree)

optimization_model.optimize()

matched_indexes = []
for v in optimization_model.getVars():
    if v.x == 1:
        print( '%s: %g ' % (v.varName,v.x))
        matched_indexes.append(int(v.varName.split('[')[1].split(']')[0]))
for i in matched_indexes:
    print(e[i])





