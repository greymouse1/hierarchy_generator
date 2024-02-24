# Import the Dataset class from dataset.py
from dataset import Dataset

# Instantiate the Dataset class
dataset = Dataset(name='ahrem_small',path='/Users/shark/Desktop/My Documents/uni/Munster/Possible_thesis/Bonn/virtual_folder/pythonProject/tri/ahrem_small',epsilon=0)

# Load data with eps=0
dataset.loadData(0)

# Get wkt in file
dataset.get_wkt()

# Get wkt unions in file
dataset.get_wkt_unions()



