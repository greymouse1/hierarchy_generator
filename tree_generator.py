import networkx as nx
from shapely.ops import unary_union
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import re

# whole function which builds tree with all nodes and surface areas
# input wkb text file with all triangles, first line are original building polygons
def treeGenerator(wkb_text_file):
    # Create an empty graph
    G = nx.DiGraph()

    # create object which will store starting polygons and their unique id's
    # this will be updated as polygons get merged

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

    initial_polygons = polygon_parser(wkb_text_file[0])

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

        # add node for current polygon
        G.add_node(node_id, geometry=geometry)
    # add AREA to data frame
    # polygon_storage['AREA'] = polygon_storage['geometry'].area

    # since first line had initial polygons, and triangles are starting from the second line
    # loop will iterate from second line to the end
    for line in wkb_text_file[1:]:

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

            # add new node for new merged polygon
            G.add_node(new_id, geometry=unary_union(filtered_polygons_geometries))
            edges = []
            for id in intersecting_polygons_ids:
                new_edge = (id,new_id)
                edges.append(new_edge)
            G.add_edges_from(edges)
            # now we update the starting storage since this one is used for checking all new triangles
            # it will have old polygons deleted and new ones added
            polygons_gdf = pd.concat([polygons_gdf,new_entry_gdf], ignore_index=True)

            # now use mask created before and invert it with tilde so all entries except the ones in the mask will be kept
            inverted_mask = ~polygons_gdf['ID'].isin(intersecting_polygons_ids)

            # filter the GeoDataFrame using the boolean mask to remove rows with specified IDs
            polygons_gdf = polygons_gdf[inverted_mask]

        else:
            next


    print(polygon_storage)
    polygon_storage.to_file("tree_polygons.shp")
    print(G.nodes)
    print(G.edges)

    # Draw the graph, remove geometry because pydot gets error if geometry is used
    # It is not important here, this graph is just for drawing
    # original graph keeps geometries
    G_without_area = nx.DiGraph()
    G_without_area.add_nodes_from(G.nodes)
    G_without_area.add_edges_from(G.edges)

    # pydot graph, fixed
    pos = nx.nx_pydot.graphviz_layout(G_without_area, prog='dot')
    nx.draw(G_without_area, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=12, font_weight='bold')
    plt.show()

    # below works well, plots tree into png
    p = nx.drawing.nx_pydot.to_pydot(G_without_area)
    p.write_png('example.png')

    print("Graph is returned")
    return G








