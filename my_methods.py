# Import the Dataset class from dataset.py
from dataset import Dataset
from tree_generator import treeGenerator, jaccardIndex
from gurobipy import *
import networkx as nx
from tqdm import tqdm

# Instantiate the Dataset class
dataset1 = Dataset(name='auerberg_atkis',path='/Users/shark/Desktop/My Documents/uni/Munster/Possible_thesis/Bonn/virtual_folder/pythonProject/tri/auerberg_atkis',epsilon=0)
dataset2 = Dataset(name='auerberg_osm',path='/Users/shark/Desktop/My Documents/uni/Munster/Possible_thesis/Bonn/virtual_folder/pythonProject/tri/auerberg_osm',epsilon=0)

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




