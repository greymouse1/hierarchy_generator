# Import the Dataset class from dataset.py
from dataset import Dataset
from tree_generator import treeGenerator

# Instantiate the Dataset class
dataset = Dataset(name='ahrem_mini',path='/Users/shark/Desktop/My Documents/uni/Munster/Possible_thesis/Bonn/virtual_folder/pythonProject/tri/ahrem_mini',epsilon=0)

# Load data with eps=0
dataset.loadData(0)

# Get wkt in file
dataset.get_wkt()

# Get wkt unions in file
dataset.get_wkt_unions()

treeGenerator(dataset.all_wkt)




