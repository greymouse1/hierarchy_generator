from shapely import wkb, wkt
from shapely.geometry import shape, MultiPolygon
from shapely.ops import unary_union
import shapefile
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os


@dataclass(init=False)
class Dataline(object):
    """docstring for Dataline"""
    x: np.ndarray
    y: np.ndarray
    diffX: np.ndarray
    diffY: np.ndarray
    name: str

    def __init__(self, x, y, name=""):
        super(Dataline, self).__init__()
        if type(x) is pd.core.series.Series:
            x = x.to_numpy()
        if type(y) is pd.core.series.Series:
            y = y.to_numpy()
        if x.shape[0] == 0:
            import pdb
            pdb.set_trace()
        self.x = x
        self.y = y
        self.x[self.y == 0] = None
        self.y[self.y == 0] = None
        self.diffX = np.diff(self.x)
        self.diffY = np.diff(self.y)
        self.name = name

    def getPositions(self, x):
        idx = np.argmax(self.x.reshape(-1, 1) >= x.reshape(1, -1), axis=0)
        idx -= 1

        y = (x - self.x[idx]) / self.diffX[idx] * self.diffY[idx] + self.y[idx]
        return y


@dataclass(init=False)
class TempArgs(object):
    """docstring for TempArgs"""
    path: str
    epsilon: str
    dataset: str
    disable_plots: bool = True
    disable_grad: bool = True
    folder: str

    def __init__(self, path=None, epsilon=None, dataset=None, folder=None):
        super(TempArgs, self).__init__()
        self.path = path if path else "../results/{}{}"
        self.epsilon = epsilon
        self.dataset = dataset
        self.disable_plots = True
        self.disable_grad = True
        self.folder = folder


def readWktFile(filename: str):
    with open(filename, "r") as f:
        f.readline()
        geom_string = f.readline()[2:]
    return wkt.loads(geom_string)


def readWkbFile(filename: str):
    if not os.path.exists(filename):
        print(filename)
        print("File is missing!")
    with open(filename, "rb") as f:
        geom = f.read()
    return wkb.loads(geom)


def readWkbAsUnion(file):
    return unary_union(readWkbFile(file))


def readShapefile(filename):
    s = shapefile.Reader(filename)
    shp = []
    #first feature of the shapefile
    for feature in s.shapeRecords():
        first = feature.shape.__geo_interface__
        shp.append(shape(first))
    shp = MultiPolygon(shp)
    return shp


def limitPolygonSize(p, limit=400000):
    #p = unary_union(p)
    if type(p) is MultiPolygon:
        return MultiPolygon([poly for poly in p.geoms if poly.area > limit])
    else:
        if p.area > limit:
            return MultiPolygon([p])
        else:
            return MultiPolygon([])