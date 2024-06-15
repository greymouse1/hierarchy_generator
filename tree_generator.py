import networkx as nx
import shapely.geometry as sg
from shapely.ops import unary_union
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import os
# whole function which builds tree with all nodes and surface areas
# input wkb text file with all triangles, first line are original building polygons
#def treeGenerator(wkb_text_file,tree_name):
class treeGenerator:
    def __init__(self,arg1,arg2):
        # Wkb text file
        self.wkb_text_file = arg1
        # Tree name
        self.tree_name = arg2
        # Tree root node
        self.tree_root = []
        # Tree leaves nodes
        self.tree_leaves = []
        # Declare empty variable which will later hold a gdf with all nodes and their geometries
        self.polygon_storage = None
        # Initialise directional graph
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

        # Grouping function takes current wkb MULTIPOLYGON line and checks if there are disjoint groups of polygons
        # inside. In case there is only one group of polygons, it outputs list with one nested list of polygons
        # If there are multiple groups, it outputs a list with nested lists, each nested list for one group of polygons
        def grouping_function(wkt_line):
            # Initialize empty list which will store groups
            groups = []

            for current_polygon in wkt_line.geoms:
                # Flag to indicate if the polygon is grouped
                grouped = False

                # If there are no existing groups, create a new group for each polygon
                if not groups:
                    groups.append([current_polygon])
                    continue  # Move to the next polygon

                # Iterate over existing groups
                for group in groups:
                    # Check if the polygon touches any polygon in the group
                    if any(current_polygon.touches(poly) for poly in group):
                        # Add the polygon to the group
                        group.append(current_polygon)
                        grouped = True
                        break

                # If the polygon is not grouped, create a new group
                if not grouped:
                    groups.append([current_polygon])

            return groups

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
        self.polygon_storage = polygons_gdf.copy()

        # add starting nodes
        # iterate over rows of the polygon_storage
        for index, row in self.polygon_storage.iterrows():
            # get Id and geometry
            node_id = row['ID']
            geometry = row['geometry']
            #append starting polygons which are later on considered leaves
            self.tree_leaves.append(self.tree_name + "_" + str(node_id))
            # add node for current polygon
            self.G.add_node(self.tree_name + "_" + str(node_id), geometry=geometry)

        # add AREA to data frame
        # self.polygon_storage['AREA'] = self.polygon_storage['geometry'].area

        # since first line had initial polygons, and triangles are starting from the second line
        # loop will iterate from second line to the end
        for line in tqdm(self.wkb_text_file[1:], desc=f"Tree {self.tree_name} is being built"):

            # get coordinates for all triangles in current line
            # now perform unary union of these triangles if there is more than one triangle
            #all_triangles = [Polygon(poly) for poly in polygon_parser(line)]
            list_of_nested_groups = grouping_function(line)
            for group in list_of_nested_groups:
                all_triangles = unary_union(group)

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
                    last_id = self.polygon_storage.iloc[-1]['ID']
                    new_id = last_id + 1
                    new_entry = {'ID': new_id, 'geometry': unary_union(filtered_polygons_geometries)}
                    new_entry_gdf = gpd.GeoDataFrame([new_entry], geometry='geometry')
                    self.polygon_storage = pd.concat([self.polygon_storage,new_entry_gdf], ignore_index=True)
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
        # This logic will check if polygons_gdf contains more than one element
        # If everything is correct, it will contain only one element and that is the root
        # If however there are more than one elements, that means we have some polygons which are
        # too far away from everything else and Delaunay triangulation for them was not performed
        # meaning that they are not connected with triangles to other polygons and can't be part of the tree
        # In order for code to work, they have to be removed from graph G and list of leaves

        # Check the length of polygons_gdf
        if len(polygons_gdf) > 1:
            # Pull IDs into a list
            polygon_ids = polygons_gdf['ID'].tolist()

            # Add tree name since IDs in the polygons_gdf are numbers
            # And detect which are erroneous polygons IDs
            modified_ids = [self.tree_name + "_" + str(id) for id in polygon_ids if self.tree_name + "_" + str(id) != self.tree_root[0]]
            print("Polygons without triangles detected")
            print("Removing excess polygons:",modified_ids)
            # Remove IDs from networkx graph G
            self.G.remove_nodes_from(modified_ids)

            # Remove IDs from tree_leaves list
            self.tree_leaves = [leaf for leaf in self.tree_leaves if leaf not in modified_ids]

        # self.polygon_storage.to_file("tree_polygons.shp") this is not needed right now

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
        #pos = nx.nx_pydot.graphviz_layout(G_without_area, prog='dot')
        #nx.draw(G_without_area, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=12, font_weight='bold')
        #plt.show()

        # below works well, plots tree into png
        p = nx.drawing.nx_pydot.to_pydot(G_without_area)
        p.write_png(f'{self.tree_name}_topography.png')

    # This function is called later on when matching process was done. Input into this is list of nodes
    # which have been matched. Function will pull leafs of all these nodes, group leaf polygons, unionise
    # them and create a buffer. This buffer is saved into .shp file. Such layer can be overlaid with starting
    # .wkt files so grouping can be visualized efficiently
    def saveShpGrouped(self, nodeList, directory_name):
        # initialise new GeoDataframe
        new_gdf = gpd.GeoDataFrame()

        for node in nodeList:
            # Since original IDs are only numbers, and IDs from matches have tree name preceding the number,
            # I have to remove the tree name
            node_id_trimmed = int(node.split("_")[1])
            my_geometry = self.polygon_storage.loc[self.polygon_storage['ID'] == node_id_trimmed, 'geometry'].convex_hull.buffer(5)
            new_geometry = {'ID': node, 'geometry': my_geometry}
            new_geometry_gdf = gpd.GeoDataFrame(new_geometry, geometry = 'geometry')
            new_gdf = pd.concat([new_gdf,new_geometry_gdf], ignore_index = True)

        if not new_gdf.empty:
            new_gdf.to_file(os.path.join(directory_name,f"{self.tree_name}_matched_nodes.shp"))
            print(f"New GeoDataFrame for {self.tree_name} matched edges was saved successfully.")
        else:
            print("No geometries found for nodes ")

        # Code for saving base polygons into shp
        base_gdf = gpd.GeoDataFrame()

        for node in self.tree_leaves:
            # Since original IDs are only numbers, and IDs from matches have tree name preceding the number,
            # I have to remove the tree name
            node_id_trimmed = int(node.split("_")[1])
            my_geometry = self.polygon_storage.loc[
                self.polygon_storage['ID'] == node_id_trimmed, 'geometry']
            new_geometry = {'ID': node, 'geometry': my_geometry}
            new_geometry_gdf = gpd.GeoDataFrame(new_geometry, geometry='geometry')
            base_gdf = pd.concat([base_gdf, new_geometry_gdf], ignore_index=True)

        if not base_gdf.empty:
            base_gdf.to_file(os.path.join(directory_name,f"{self.tree_name}_base_nodes.shp"))
            print(f"New GeoDataFrame for {self.tree_name} base nodes was saved successfully.")
        else:
            print("No geometries found for base nodes ")
def jaccardIndex(tree1,tree2):

    # create new empty graph
    # this graph will hold edges between two trees and their weights/Jaccard indexes
    G = nx.Graph()

    # since the code doesn't know which node of a tree is root node, it has to pull it out, for each tree
    tree1_root = tree1.tree_root[0]
    tree2_root = tree2.tree_root[0]
    print("Root 1",tree1_root)
    print("Root 2", tree2_root)

    # Precompute shapes and second tree successors for each node
    geometry1_shp = {node_id: sg.shape(data['geometry']) for node_id, data in tree1.G.nodes(data=True)}
    geometry2_shp = {node_id: sg.shape(data['geometry']) for node_id, data in tree2.G.nodes(data=True)}
    tree2_successors = {node_id: list(tree2.G.successors(node_id)) for node_id in tree2.G.nodes()}

    # Compute nodes of first tree in advance
    dfs_preorder_nodes_tree1 = list(nx.dfs_preorder_nodes(tree1.G, source=tree1_root))

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
    # as edges between trees are checked and Jaccard Index is calculated, resulting edges with JI are packed
    # into a new graph

    # Edges to be added
    edges_to_add = []

    for node_id in tqdm(dfs_preorder_nodes_tree1,desc="Calculating Jaccard Index between nodes"):
        #order nodes in dfs order
        geometry1 = geometry1_shp[node_id] # pull geometry from current node from tree 1
        node1 = node_id
        #G.add_node(node1,bipartite=0)

        stack = [tree2_root]
        visited = set()

        while stack:
            current_node = stack.pop()
            node2 = current_node
            #G.add_node(node2,bipartite=1)
            visited.add(current_node)
            geometry2 = geometry2_shp[current_node] # pull geometry from current node from tree 2

            # First condition to be checked is if there is existing edge between node from T1 and T2
            # If there is, it can only be 0 and in that case we disregard whole subtree for rooted at current
            # node in T2, and proceed to the next available node

            #if G.has_edge(node1,node2):
            #    print("edge 0 detected, avoid subtree rooted here, move to next node")
            #    continue

            # If there is no edge, code will continue with calculation of intersection as usual

            # Try creating intersection
            intersection = geometry1.intersection(geometry2).area

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
            union = geometry1.union(geometry2).area
            jaccard_index = intersection / union
            edges_to_add.append((node1,node2,jaccard_index))
            #G.add_edge(node1,node2,weight=jaccard_index)

            # Iterate through children and add them to stack
            tree2_children = tree2_successors[current_node]
            for child in tree2_children:
                if child not in visited:
                    stack.append(child)

    # Adding whole list of weighted edges outside the loop
    G.add_weighted_edges_from(edges_to_add)

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








