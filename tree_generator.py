import networkx as nx
import shapely.geometry as sg
from shapely.ops import unary_union
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from rtree import index
from bidict import bidict

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
        # Dictionary which will store all unique shortest paths from root to each leaf
        self.leaf_paths = {}
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
            self.G.add_node(self.tree_name + "_" + str(node_id), geometry=geometry, leaves=[self.tree_name + "_" + str(node_id)])

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
                adjusted_id = self.tree_name + "_" + str(new_id)
                # pack all leaves from child nodes into a list
                leafs_list = []
                for _ in intersecting_polygons_ids:
                    leafs_list.extend(self.G.nodes[self.tree_name + "_" + str(_)]['leaves'])
                self.G.add_node(adjusted_id, geometry=unary_union(filtered_polygons_geometries), leaves = leafs_list )

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

        # polygon_storage.to_file("tree_polygons.shp") this is not needed right now

    def calculateLeafs(self):
        for i in range(len(self.tree_leaves)):
            leaf = self.tree_leaves[i]
            shortest_path = nx.shortest_path(self.G, source=self.tree_root[0], target=leaf)
            self.leaf_paths[leaf] = shortest_path

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

    # Precomupte unions of leaf polygons for each node in T1 and T2

    # Precompute shapes and second tree successors for each node
    geometry1_shp = {node_id: sg.shape(tree1.G.nodes[node_id]['geometry']) for node_id in tree1.tree_leaves}
    geometry2_shp = {node_id: sg.shape(tree2.G.nodes[node_id]['geometry']) for node_id in tree2.tree_leaves}

    # Precompute nodes and their leaves so graph isn't being accessed in the loop later
    tree1_leaves = {node_id: list(tree1.G.nodes[node_id]['leaves']) for node_id in tree1.G.nodes()}
    tree2_leaves = {node_id: list(tree2.G.nodes[node_id]['leaves']) for node_id in tree2.G.nodes()}

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
    # as edges between trees are checked and Jaccard Index is calculated, resulting deges with JI are packed
    # into a new graph

    # Edges to be added for the new graph
    edges_to_add = []

    # Create a bidirectional mapping between string ID and integer ID
    string_to_integer_mapping = bidict()

    # Construct rtree index
    tree2_index = index.Index()
    for i,node in enumerate(tree2.tree_leaves):
        string_to_integer_mapping[node] = i

        # Pull coordinates of the polygon for each node
        polygon = tree2.G.nodes[node]["geometry"]

        # Calculate bounding box of a node
        bbox = polygon.bounds

        # Add node with its ID and bounding box to the rtree index
        tree2_index.insert(i,bbox)

    processed_edges = set()
    leaf_inst_union = {} # hold intersections and unions for each edge between leaves of two trees
    for node_id in tqdm(tree1.tree_leaves, desc="Jaccard Index is being calculated"):
        # Pull node geometry and ID
        node1_geometry = geometry1_shp[node_id] # pull geometry from current node from tree 1
        node1_id = node_id

        # Create bbox for node1
        node1_bbox = node1_geometry.bounds

        # Query the rtree index to find intersections
        potential_intersections = list(tree2_index.intersection(node1_bbox))

        if len(potential_intersections) > 0:
            # Pull all nodes for leaf in T1
            unique_nodes_t1 = tree1.leaf_paths[node1_id]
            starting_set = set([])
            # if there is really intersection between leafs, add them to the dictionaryu
            for i in potential_intersections:
                node2_id = string_to_integer_mapping.inv[i]
                if geometry1_shp[node_id].intersects(geometry2_shp[node2_id]):
                    intersection = geometry1_shp[node_id].intersection(geometry2_shp[node2_id]).area
                    union = geometry1_shp[node_id].union(geometry2_shp[node2_id]).area
                    leaf_inst_union[node_id+"-"+node2_id] = (intersection,union)
                    # Since matching is done in a one-to-many manner
                    # the second set will have to be cleaned of redundant nodes
                    # since shortest paths for two or more nodes in T2 may overlap
                    starting_set = starting_set.union(set(tree2.leaf_paths[node2_id]))

            for x in unique_nodes_t1:
                for y in list(starting_set):
                    # Concatenate x and y to form a unique edge identifier
                    edge_id = f"{x}-{y}"
                    if edge_id in processed_edges:
                        continue
                    edges_to_add.append((x, y,"0"))
                    # Mark the edge as processed
                    processed_edges.add(edge_id)
        else:
            continue
    # Adding whole list of weighted edges outside the loop
    G.add_weighted_edges_from(edges_to_add)
    for u,v in tqdm(G.edges):
        leaves_in_t1 = tree1_leaves[u]
        leaves_in_t2 = tree2_leaves[v]
        sum_int = 0
        sum_uni = 0
        for l1 in leaves_in_t1:
            for l2 in leaves_in_t2:
                edge = l1 + "-" + l2
                if edge in leaf_inst_union.keys():
                    sum_int = sum_int + leaf_inst_union[edge][0]
        ji = sum_int/(tree1.G.nodes[u]['geometry'].union(tree2.G.nodes[v]['geometry'])).area
        G[u][v]['weight'] = ji

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








