import networkx as nx
import shapely.geometry as sg
from shapely.ops import unary_union
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

# whole function which builds tree with all nodes and surface areas
# input wkb text file with all triangles, first line are original building polygons
#def treeGenerator(wkb_text_file,tree_name):
class treeGenerator:
    def __init__(self,arg1,arg2):
        self.wkb_text_file = arg1
        self.tree_name = arg2
        self.tree_root = []
        self.tree_leaves = []
        self.G = nx.DiGraph()
        # List which will store all unique shortest paths from root to each leaf
        self.leaf_list = []
        # Call method to run all calculations when instance of a class is initialized
        self.run_calculations()
        # Call method to calculate shortest paths to each leaf, this has to be done after run_calculations
        # because graph has to be finalised
        self.calculateLeafs()

    # create object which will store starting polygons and their unique id's
    # this will be updated as polygons get merged
    def run_calculations(self):
        def polygon_parser(input_line):
            polygons = []
            input_line = str(input_line)
            input_line = re.split(r"\)\), \(\(|\), \(", input_line) # split by polygons and nested polygons
            for polygon_str in input_line:
                # remove unnecessary characters
                polygon_str = polygon_str.replace("MULTIPOLYGON (((", "").replace(")))", "").replace(")), ((", ")),((")
                # extract coordinates
                coords = polygon_str.split(",")
                polygon_coords = [tuple(map(float, coord.strip().split())) for coord in coords if coord.strip()]
                polygons.append(polygon_coords)
            return polygons

        initial_polygons = polygon_parser(self.wkb_text_file[0])

        # store list of polygon coordinates
        polygon_shapes = [Polygon(poly) for poly in initial_polygons]
        # store a list of unique ID's for every polygon
        polygon_ids = [i for i, _ in enumerate(initial_polygons)]

        # create geopandas with all polygons geometries and ID's
        # now each polygon has unique ID and coordinates in this object
        # this storage will constantly be updated as tree is growing
        # when merge occurs, old polygons will be deleted and new ones added
        polygons_gdf = gpd.GeoDataFrame({'ID': polygon_ids, 'geometry': polygon_shapes})

        # create storage which will hold all polygons, old and new parent ones with their
        # respective IDs and surface areas
        polygon_storage = polygons_gdf.copy()

        # add starting nodes
        # iterate over rows of the polygon_storage
        for index, row in polygon_storage.iterrows():
            # get Id and geometry
            node_id = row['ID']
            geometry = row['geometry']
            #append starting polygons which are later on considered leaves
            self.tree_leaves.append(self.tree_name + "_" + str(node_id))
            # add node for current polygon
            self.G.add_node(self.tree_name + "_" + str(node_id), geometry=geometry)

        # add AREA to data frame
        # polygon_storage['AREA'] = polygon_storage['geometry'].area

        # since first line had initial polygons, and triangles are starting from the second line
        # loop will iterate from second line to the end
        for line in tqdm(self.wkb_text_file[1:], desc=f"Tree {self.tree_name} is being built"):

            # get coordinates for all triangles in current line
            # now perform unary union of these triangles if there is more than one triangle
            all_triangles = [Polygon(poly) for poly in polygon_parser(line)]
            all_triangles = unary_union(all_triangles)

            # check if coordinates of this triangle/polygon are shared with
            # some of the polygons
            intersecting_polygons_ids = []
            for index, row in polygons_gdf.iterrows():
                polygon_id = row['ID']
                polygon_geometry = row['geometry']
                if all_triangles.intersects(polygon_geometry):
                    intersecting_polygons_ids.append(polygon_id)

            #if intersecting_polygons_ids:
                #print("Polygons from the second group that intersect with the polygon in the first group:")
                #print(intersecting_polygons_ids)
            #else:
                #print("No polygon from the second group intersects with the polygon in the first group")

            if len(intersecting_polygons_ids) > 1:
                # Clear root holder since new parent is detected
                self.tree_root.clear()
                # Create a boolean mask to filter rows with IDs present in the list
                mask = polygons_gdf['ID'].isin(intersecting_polygons_ids)
                filtered_polygons = polygons_gdf[mask]
                # Now extract geometries into a list
                filtered_polygons_geometries = filtered_polygons['geometry'].tolist()
                triangles_and_polygons = unary_union([all_triangles, unary_union(filtered_polygons_geometries)])
                #print("Triangles and polygons union")
                #print(triangles_and_polygons)
                # now storage with original polygons needs to be updated for changes
                # concatenate the IDs from the list, this storage keeps original polygons
                # no triangles are merged here, just original polygons go in this storage
                # even when "merged" poligon is considered it will still have area of only
                # original polygons without the triangles
                # new_id = '_'.join(str(id_) for id_ in intersecting_polygons_ids)
                last_id = polygon_storage.iloc[-1]['ID']
                new_id = last_id + 1
                new_entry = {'ID': new_id, 'geometry': unary_union(filtered_polygons_geometries)}
                new_entry_gdf = gpd.GeoDataFrame([new_entry], geometry='geometry')
                polygon_storage = pd.concat([polygon_storage,new_entry_gdf], ignore_index=True)
                #print("success")

                # log current node, once code comes to end, this will hold root
                self.tree_root.append(self.tree_name + "_" + str(new_id))
                # add new node for new merged polygon
                self.G.add_node(self.tree_name + "_" + str(new_id), geometry=unary_union(filtered_polygons_geometries))
                edges = []
                for id in intersecting_polygons_ids:
                    new_edge = (self.tree_name + "_" + str(new_id),self.tree_name + "_" + str(id))
                    edges.append(new_edge)
                self.G.add_edges_from(edges)
                # now we update the starting storage since this one is used for checking all new triangles
                # it will have old polygons deleted and new ones added
                polygons_gdf = pd.concat([polygons_gdf,new_entry_gdf], ignore_index=True)

                # now use mask created before and invert it with tilde so all entries except the ones in the mask will be kept
                inverted_mask = ~polygons_gdf['ID'].isin(intersecting_polygons_ids)

                # filter the GeoDataFrame using the boolean mask to remove rows with specified IDs
                polygons_gdf = polygons_gdf[inverted_mask]

            else:
                next
        # polygon_storage.to_file("tree_polygons.shp") this is not needed right now

    def calculateLeafs(self):
        for i in range(len(self.tree_leaves)):
            self.leaf_list.append(nx.shortest_path(self.G, source=self.tree_root[0], target=self.tree_leaves[i]))

    def drawGraph(self):
        # Draw the graph, remove geometry because pydot gets error if geometry is used
        # It is not important here, this graph is just for drawing
        # original graph keeps geometries
        G_without_area = nx.DiGraph()
        G_without_area.add_nodes_from(self.G.nodes)
        G_without_area.add_edges_from(self.G.edges)

        # pydot graph, fixed
        pos = nx.nx_pydot.graphviz_layout(G_without_area, prog='dot')
        nx.draw(G_without_area, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=12, font_weight='bold')
        plt.show()

        # below works well, plots tree into png
        p = nx.drawing.nx_pydot.to_pydot(G_without_area)
        p.write_png(f'{self.tree_name}_topography.png')

def jaccardIndex(tree1,tree2):

    # create new empty graph
    # this graph will hold edges between two trees and their weights/Jaccard indexes
    G = nx.Graph()

    # since the code doesn't know which node of a tree is root node, it has to pull it out, for each tree
    tree1_root = tree1.tree_root[0]
    tree2_root = tree2.tree_root[0]
    print("Root 1",tree1_root)
    print("Root 2", tree2_root)

    # Populate new graph
    # Add nodes from graph1 with bipartite=0
    G.add_nodes_from([(node, {"bipartite": 0}) for node in tree1.G.nodes])

    # Add nodes from graph2 with bipartite=1
    G.add_nodes_from([(node, {"bipartite": 1}) for node in tree2.G.nodes])

    # now I need whole topology of each tree below root
    # subtree_tree1 = nx.bfs_tree(tree1, tree1_root)
    # subtree_tree2 = nx.dfs_tree(tree2, tree2_root)

    # now is start iterating; I check each node from tree1 with every node from tree 2, from root to leaves
    # if there is a node in tree 2 for which intersection is 0, then whole subtree of that node is removed from tree 2
    # iterator, so children wouldn't be checked in the following iterations (since if intersection with parent is 0, that
    # means intersection with every child will also be 0
    # as edges between trees are checked and Jaccard Index is calculated, resulting deges with JI are packed
    # into a new graph

    for node_id in tqdm(nx.dfs_preorder_nodes(tree1.G,source=tree1_root),desc="Calculating Jaccard Index between nodes"): #order nodes in dfs order
        geometry1 = tree1.G.nodes[node_id].get('geometry') # pull geometry from current node from tree 1
        node1 = node_id
        #G.add_node(node1,bipartite=0)

        stack = [tree2_root]
        visited = set()

        while stack:
            current_node = stack.pop()
            node2 = current_node
            #G.add_node(node2,bipartite=1)
            visited.add(current_node)
            geometry2 = tree2.G.nodes[current_node].get('geometry') # pull geometry from current node from tree 2

            # First condition to be checked is if there is existing edge between node from T1 and T2
            # If there is, it can only be 0 and in that case we disregard whole subtree for rooted at current
            # node in T2, and proceed to the next available node

            if G.has_edge(node1,node2):
                print("edge 0 detected, avoid subtree rooted here, move to next node")
                continue

            # If there is no edge, code will continue with calculation of intersection as usual

            # Convert geometries to Shapely objects
            geometry1_shp = sg.shape(geometry1)
            geometry2_shp = sg.shape(geometry2)
            intersection = geometry1_shp.intersection(geometry2_shp).area

            # Check condition for the current node
            # If intersection is 0 for the current pair, get subtrees rooted in them and populated all the edges with
            # weight 0
            if intersection == 0:
                #print(f"Nodes {node1} and {node2} aren't intersecting")
                #current_subtree_T1 = nx.dfs_preorder_nodes(tree1.G,node_id)
                #current_subtree_T2 = nx.dfs_preorder_nodes(tree2.G,current_node)
                ##G.add_edges_from([(node1, "T2_" + str(target_node), {'weight': 0}) for target_node in current_subtree])
                #G.add_edges_from([(u,v, {'weight': 0}) for u in current_subtree_T1 for v in current_subtree_T2])
                continue  # Skip subtree if condition not satisfied

            # If intersection is != 0 then calculate weight
            # create weight as Jaccard Index
            union = geometry1_shp.union(geometry2_shp).area
            jaccard_index = intersection / union
            G.add_edge(node1,node2,weight=jaccard_index)

            # Iterate through children and add them to stack
            for child in tree2.G.successors(current_node):
                if child not in visited:
                    stack.append(child)

    # Code for plotting which is not really necessary
    #for u, v, data in G.edges(data=True):
    #    weight = data.get('weight', None)
    #    print(f"Edge ({u}, {v}) has weight {weight}")

    #for u, v, attrs in G.edges('T2_31', data=True):
    #    print(f"Edge from {u} to {v} with attributes: {attrs}")

    # Generate lists of nodes based on their bipartite attribute
    #nodes_bipartite_0 = [n for n, d in G.nodes(data=True) if d['bipartite'] == 0]

    # Generate layout for visualization
    #pos = nx.bipartite_layout(G, nodes_bipartite_0)

    # Draw the graph
    #nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12, font_weight='bold')
    #plt.title('Bipartite Graph Layout')
    #plt.show()
    return G








