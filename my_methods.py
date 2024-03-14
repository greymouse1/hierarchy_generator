# Import the Dataset class from dataset.py
from dataset import Dataset
from tree_generator import treeGenerator, jaccardIndex

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

jaccardIndex(T1,T2)