# Import the Dataset class from dataset.py
from dataset import Dataset
from tree_generator import treeGenerator, jaccardIndex
import pickle
import networkx as nx
import datetime
import os

# Test dataset
dataset1 = Dataset(name='dummy',path='/Users/shark/Desktop/dontsync.nosync/thesis/hierarchy_generator/tri/dummy',epsilon=0)
dataset2 = Dataset(name='dummy2',path='/Users/shark/Desktop/dontsync.nosync/thesis/hierarchy_generator/tri/dummy2',epsilon=0)

# Load data with eps=0
dataset1.loadData(0)
dataset2.loadData(0)


# Get wkt unions in file - not really necessary anymore
# dataset.get_wkt_unions()

# Generate tree
# Right now I can use "dataset" class instance from above, I could as well
# use .txt file directly which was created with dataset.get_wkt()
# insert tree name after file name
T1 = treeGenerator(dataset1.all_wkt,"T1")
T2 = treeGenerator(dataset2.all_wkt,"T2")

# Create new timestamped folder for pngs, pkls and graphml
def create_timestamped_folder(dataset_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{dataset_name}_{timestamp}"
    folder_path = os.path.join("trees/",folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

new_folder_path = create_timestamped_folder("dummy")

# Get wkt in file
dataset1.get_wkt(new_folder_path)
dataset2.get_wkt(new_folder_path)

# Draw png
T1.drawGraph(new_folder_path)
T2.drawGraph(new_folder_path)

# Save trees as pickles
T1_pkl_path = os.path.join(new_folder_path, "T1.pkl")
T2_pkl_path = os.path.join(new_folder_path, "T2.pkl")

with open(T1_pkl_path, "wb") as f:
    pickle.dump(T1, f)
with open(T2_pkl_path, "wb") as f:
    pickle.dump(T2, f)

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
# ignored and only vertices taken into account when jaccardIndex is performed
weighted_graph = jaccardIndex(T1,T2)
graphml_path = os.path.join(new_folder_path,"jaccard_index.graphml")
nx.write_graphml(weighted_graph,graphml_path )
print(weighted_graph)


