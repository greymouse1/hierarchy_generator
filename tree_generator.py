import networkx as nx
from shapely.ops import unary_union
from shapely.geometry import Polygon
import geopandas as gpd

# whole function which builds tree with all nodes and surface areas
# input wkb text file with all triangles, first line are original building polygons
def treeGenerator(wkb_text_file):

    # create object which will store starting polygons and their unique id's
    # this will be updated as polygons get merged

    def polygon_parser(input_line):
        polygons = []
        for polygon_str in input_line.split(")), (("):
            # remove unnecessary characters
            polygon_str = polygon_str.replace("MULTIPOLYGON (((", "").replace(")))", "").replace(")), ((", ")),((")
            # extract coordinates
            coords = polygon_str.split(",")
            polygon_coords = [tuple(map(float, coord.strip().split())) for coord in coords if coord.strip()]
            polygons.append(polygon_coords)
            return polygons

    initial_polygons = polygon_parser(wkb_text_file[0])

    # store list of polygon coordinates
    polygon_shapes = [Polygon(poly[0]) for poly in initial_polygons]
    # store a list of unique ID's for every polygon
    polygon_ids = [i for i in enumerate(initial_polygons)]

    # create geopandas with all polygons geometries and ID's
    # now each polygon has unique ID and coordinates in this object
    polygons_gdf = gpd.GeoDataFrame({'ID': polygon_ids, 'geometry': polygon_shapes})

    # since first line had initial polygons, and triangles are starting from the second line
    # loop will iterate from second line to the end
    for line in wkb_text_file[1:]:

        # get coordinates for all triangles in current line
        all_triangles = polygon_parser(line)

        # now perform unary union of these triangles
        # check if coordinates of this triangle/polygon are shared with
        # some of the polygons






