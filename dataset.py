import pandas as pd
from functools import partial, wraps
import os
from utils import readWkbFile, readShapefile
from shapely.ops import unary_union
from shapely import wkb
from plot import drawPoly
from utils import limitPolygonSize
from shapely.geometry import Polygon, MultiPolygon
import matplotlib
from matplotlib import pyplot as plt
import math
from time import time

import pdb


class Dataset(object):
    """docstring for Dataset"""

    def __init__(self, name, path, epsilon, gt_epsilon=1e-6):
        super(Dataset, self).__init__()
        self.name = name
        self.data = {}
        self.gt_eps = gt_epsilon
        self.all_wkt = []  # New instance variable to hold individual WKT polygons
        self.wkt_union = [] # New instance variable to hold union of current and previous WKT polygons

        cases = [epsilon, self.gt_eps] if epsilon != 0 else range(1)
        for eps in cases:
            eps_part = f"_{eps:.6f}" if eps != 0 else ""
            self.data[eps] = {}
            self.data[eps]["eps"] = eps
            self.data[eps]["path"] = path.format(self.name, eps_part)
            self.data[eps]["data"] = None
            self.data[eps]["path_table"] = self.data[eps][
                "path"] + f"/{self.name}_results.csv"
            self.data[eps]["data_table"] = None
            self.data[eps]["unionPolys"] = None

        self.path_gt = f"~/sciebo/Forschung/adopt-merge/data/{name}/{name}_gt.shp"
        self.shapefile_gt = None

    def __lt__(self, other):
        return self.name < other.name
#-------------------------------------------------------------------------------
    def get_wkt(self, output_folder=None): # Writes wkt files all together into one test file
        if not hasattr(self, 'all_wkt') or not self.all_wkt:
            print("No WKT polygons stored.")
            return
        if output_folder is None:
            output_folder = os.getcwd()
        output_file = os.path.join(output_folder, 'all_wkt_polygons.txt')
        with open(output_file, 'w') as f:
            for polygon in self.all_wkt:
                f.write(f"{polygon}\n")
        print("WKT polygons have been written to file.")
        print("File path:", output_file)

    def get_wkt_unions(self, output_folder=None): # Writes unionised wkt files into one text file
        if not hasattr(self, 'wkt_union') or not self.wkt_union:
            print("No WKT polygons stored.")
            return
        if output_folder is None:
            output_folder = os.getcwd()
        output_file = os.path.join(output_folder, 'wkt_union_polygons.txt')
        with open(output_file, 'w') as f:
            for polygon in self.wkt_union:
                f.write(f"{polygon}\n")
        print("WKT unionised polygons have been written to file.")
        print("File path:", output_file)

    def resultsLoaded(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            eps = args[0]
            if eps in self.data and self.data[eps]["data_table"] is None:
                print(
                    f"Loading table data for dataset {self.name:>15s} with epsilon: {eps:8.06f}"
                )
                self.data[eps]["data_table"] = readResults(
                    self.data[eps]["path_table"])
            return func(self, *args, **kwargs)

        return wrapper

    def shapefileLoaded(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.shapefile_gt is None:
                if os.path.exists(os.path.expanduser(self.path_gt)):
                    print(f"Loading shapefile for dataset {self.name:>16s}")
                    self.shapefile_gt = readShapefile(
                        os.path.expanduser(self.path_gt))
            else:
                print(f"Can not load shapefile for dataset {self.name:>16s}. " + \
                    "Reason: File does not exist.")
            return func(self, *args, **kwargs)

        return wrapper

    def wkbLoaded(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            eps = args[0]
            if self.data[eps]["unionPolys"] is None:
                print(
                    f"Loading WKB data for dataset {self.name:>17s} with epsilon: {eps:8.06f}"
                )
                readSquentialWkb(self, eps)
                number = len(self.data[eps]["unionPolys"])
                print(f"{number:5d} WKB files loaded.")

            return func(self, *args, **kwargs)

        return wrapper

    def wkbUnload(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            self.data[args[0]]["unionPolys"] = None
            #self.data[args[0]]["polys"] = None
            self.data[args[0]]["filenames"] = None
            return result

        return wrapper

    @resultsLoaded
    @shapefileLoaded
    @wkbLoaded
    def loadData(self, eps):
        pass

    @resultsLoaded
    def getResultsWithMetrics(self,
                              eps,
                              shapefile_gt=None,
                              force_compute=False):
        if not eps in self.data:
            return None

        metrics = ["jaccard", "simA", "hausdorff", "extEdges", "intEdges"]
        if not all(
                hasattr(self.data[eps]["data_table"], attr)
                for attr in metrics) or force_compute:
            self.addMetrics(eps, metrics, shapefile_gt, force_compute)
            self.data[eps]["data_table"].to_csv(self.data[eps]["path_table"],
                                                index=False)
        if not hasattr(self.data[eps]["data_table"], "idx") or force_compute:
            self.data[eps]["data_table"]["idx"] = [
                i for i in range(len(self.data[eps]["data_table"]))
            ]
        return self.data[eps]["data_table"]

    @resultsLoaded
    @wkbLoaded
    @shapefileLoaded
    def addMetrics(self, eps, metrics, shapefile_gt=None, force_compute=False):
        from evaluation import computeMetrics
        from tqdm.contrib.concurrent import process_map
        files = self.data[eps]["unionPolys"]
        table = self.data[eps]["data_table"]
        if not hasattr(table, "area") or force_compute:
            table["area"] = [p.area for p in files]
        if not hasattr(table, "perimeter") or force_compute:
            table["perimeter"] = [p.length for p in files]
        if not hasattr(table, "extEdges") or not hasattr(
                table, "intEdges") or force_compute:
            extEdges = []
            intEdges = []

            def countPoly(poly):
                if type(poly) is MultiPolygon:
                    e = 0
                    i = 0
                    for p in poly.geoms:
                        u, v = countPoly(p)
                        e += u
                        i += v
                else:
                    e = len(poly.exterior.coords)
                    i = 0
                    for ring in poly.interiors:
                        i += len(ring.coords)
                return e, i

            for p in files:
                e, i = countPoly(p)
                extEdges.append(e)
                intEdges.append(i)
            table["extEdges"] = extEdges
            table["intEdges"] = intEdges

        if (not hasattr(table, "jaccard")
                or force_compute) and (self.shapefile_gt != None
                                       or shapefile_gt != None):
            evaluate = partial(computeMetrics,
                               polyB=self.shapefile_gt
                               if shapefile_gt is None else shapefile_gt)
            files = [limitPolygonSize(f) for f in files]
            jaccard, hausdorff, A = zip(
                *process_map(evaluate, files, chunksize=1))
            res = {"jaccard": jaccard, "simA": A, "hausdorff": hausdorff}
            for key, value in res.items():
                table[key] = value if not hasattr(
                    table, key) or force_compute else table[key]
        self.data[eps]["data_table"] = table

    def findBest(self, eps, metric):
        data = self.getResultsWithMetrics(eps)
        if metric == "hausdorff":
            data_new = data[data["hausdorff"] != 0]
            # Select all values with minimum hausdorff distance -- array with true/false
            mins = data_new.hausdorff == data_new.hausdorff.iloc[
                data_new.hausdorff.argmin()]
            # Index of minimum hausdorff distance with maximum jaccard index
            # data_new.jaccard[mins] is new short table
            idx = data_new.jaccard[mins].argmax()
            # Table of min jaccard values
            tab = data_new[mins]
            # Get the entry with jac iou and min hausdorff
            b = tab.iloc[idx]
        else:
            idx = data[metric].argmax()
            b = data.iloc[idx]
            #if metric == "simA":
            #  print(b)
        return int(b.idx)

    @resultsLoaded
    @wkbLoaded
    def saveBest(self, eps, metric):
        file = self.data[eps]["path"] + f"/{metric}.bin"
        poly = self.data[eps]["unionPolys"][self.findBest(eps, metric)]
        with open(file, "wb") as f:
            f.write(wkb.dumps(poly))

    def getBest(self, eps, metric, force_compute=False):
        if not eps in self.data:
            return None
        file = self.data[eps]["path"] + f"/{metric}.bin"
        if not os.path.exists(file) or force_compute:
            self.saveBest(eps, metric)
        return readWkbFile(file)

    def printBestValues(self, eps):
        if not eps in self.data:
            print(f"Epsilon {eps} not loaded. Can not be printed.")
            return
        metrics = ["jaccard", "simA", "hausdorff", "extEdges", "intEdges"]
        idxs = [self.findBest(eps, metric) for metric in metrics[:3]]
        print(idxs)
        max_values = self.data[eps]["data_table"].iloc[idxs]
        sorted_vals = max_values.sort_values(max_values.columns[0])
        print(sorted_vals[[max_values.columns[0], *metrics]])

    @wkbUnload
    def getDataLine(self, eps):
        from utils import Dataline
        data = self.getResultsWithMetrics(eps).sort_values("alpha")
        line = Dataline(data.alpha, data.jaccard, self.name)
        return line

    @resultsLoaded
    @wkbLoaded
    @shapefileLoaded
    def plotGradient(self, eps, rotate=0, fontSize=8.5):
        from gradient import AnchoredHScaleBar
        from matplotlib.collections import PatchCollection

        polys = self.data[eps]["unionPolys"]
        fig, axs = plt.subplots()
        if self.name == "selection":
            fig.set_size_inches(1.6, 1.6)
        elif self.name == "ahrem":
            fig.set_size_inches(4.5, 4.5)
        else:
            fig.set_size_inches(6, 6)

        font = {'size': fontSize, 'family': 'serif'}
        matplotlib.rc('font', **font)
        matplotlib.rc('text', usetex=True)

        axs.set_aspect('equal')
        plt.axis('off')

        plt.subplots_adjust(top=1,
                            bottom=0,
                            right=1,
                            left=0,
                            hspace=0,
                            wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        axs.set_xlim(polys[0].bounds[0::2])
        axs.set_ylim(polys[0].bounds[1::2])
        patches = []
        cmap = plt.get_cmap("gray")
        # Starting with big areas, then smaller ones
        for i, poly in enumerate(polys[::-1]):
            c = (len(polys) - 1 - i) / (len(polys) * 1.1 - 1)
            print(f"Index: {i:4}, grey scale color: {c:7.5f}")
            patches.extend(
                drawPoly(poly, fc=cmap(c), lw=0, ec="None", rotate=rotate))

        # For displaying single regions
        #i = 4
        #c = (len(polys) - 1 - i) / (len(polys) * 1.1 - 1)
        #print(c)
        #patches.extend(
        #    drawPoly(unary_union(polys[:-i]),
        #             fc=cmap(c),
        #             lw=0,
        #             ec="None",
        #             rotate=rotate))

        patches.extend(
            drawPoly(self.shapefile_gt,
                     fc="none",
                     ec="g",
                     lw=1.5,
                     rotate=rotate))
        if self.gt_eps != None and self.gt_eps in self.data and hasattr(
                self.data[eps]["data_table"], "jaccard"):
            patches.extend(
                drawPoly(limitPolygonSize(self.getBest(self.gt_eps, "jaccard")),
                         fc="none",
                         ec="r",
                         lw=1.5,
                         rotate=rotate))
        if self.gt_eps != None and self.gt_eps in self.data and hasattr(
                self.data[eps]["data_table"], "jaccard"):
            patches.extend(
                drawPoly(limitPolygonSize(self.getBest(self.gt_eps,
                                                       "hausdorff")),
                         fc="none",
                         ec="b",
                         lw=1.5,
                         rotate=rotate,
                         linestyle=(0, (5, 5))))
        collection = PatchCollection(patches, match_original=True)
        axs.add_collection(collection)

        dx = polys[0].bounds[2] - polys[0].bounds[0]
        dy = polys[0].bounds[3] - polys[0].bounds[1]
        length = 0.25
        width = 0.015
        size = fig.get_size_inches()
        if dx > dy:
            width *= dx / dy
            fig.set_size_inches(size[0], size[1] * dy / dx)
        elif dy > dx:
            length *= dy / dx
            fig.set_size_inches(size[0] * dx / dy, size[1])
        if dx <= 1e4:
            scale_width = round(dx / 1e3) * 1e2
            unit = f"{round(scale_width)} m"
        elif dx <= 1e5:
            scale_width = round(dx / 1e4) * 1e3
            unit = f"{round(scale_width/1000)} km"
        elif dx <= 1e6:
            scale_width = round(dx / 1e5) * 1e4
            unit = f"{round(scale_width/1000)} km"
        print(
            "Gradient figure size (inches):",
            f"{fig.get_size_inches()[0]:5.3f} x {fig.get_size_inches()[1]:5.3f}"
        )
        ob = AnchoredHScaleBar(
            size=scale_width,
            label=f"{unit}",
            loc=1 if self.name != "gruppe1" else 4,
            frameon=False,
            pad=0.4,
            extent=0.1 / fig.get_size_inches()[1],
            sep=2,
            linekw=dict(color="black", lw=0.75),
        )
        plt.gca().add_artist(ob)

        filename = self.data[eps]["path"] + "/gradient.pdf"
        fig.savefig(filename, bbox_inches='tight', pad_inches=0)

#-------------------------------------------------------------------------------

    @resultsLoaded
    def plotJaccard(self,
                    eps,
                    metrics=["jaccard", "simA", "hausdorff"],
                    linewidth=1):
        fig = plt.figure()
        a, b = fig.get_size_inches()
        fig.set_size_inches(6, 1.8)
        ax1 = plt.gca()

        plot_data = self.data[eps]["data_table"]
        lns = []
        if "jaccard" in metrics:
            lns += ax1.plot(plot_data.alpha[plot_data.jaccard > 0],
                            plot_data.jaccard[plot_data.jaccard > 0],
                            "o",
                            label=r"Jaccard Similarity $IoU$",
                            markersize=1,
                            linewidth=linewidth,
                            color="red")
        if "simA" in metrics:
            lns += ax1.plot(plot_data.alpha[plot_data.simA > 0],
                            plot_data.simA[plot_data.simA > 0],
                            linestyle="dashed",
                            label=r"Area Similarity $V_\mathrm{A}$",
                            linewidth=linewidth,
                            color="orange")
        if "simP" in metrics:
            lns += ax1.plot(plot_data.alpha[plot_data.simP > 0],
                            plot_data.simP[plot_data.simP > 0],
                            linestyle="dotted",
                            label=r"Perimeter Similarity $V_\mathrm{P}$",
                            linewidth=linewidth,
                            color="purple")

        ax1.grid(True)
        #ax1.set_title("Jaccard Index in relation to alpha-value")
        ax1.set_xlabel(r"$\lambda$")
        ax1.set_ylabel("Similarity")
        ax1.set_xlim([0, 0.12])
        ax1.set_ylim([0, 1])

        if "hausdorff" in metrics:
            ax2 = ax1.twinx()
            lns += ax2.plot(plot_data.alpha[plot_data.hausdorff > 0],
                            plot_data.hausdorff[plot_data.hausdorff > 0],
                            linestyle="dashdot",
                            label=r"Hausdorff distance $d_\mathrm{H}$",
                            linewidth=linewidth,
                            color="blue")
            ax2.set_ylabel("Hausdorff Distance [m]")
            ax2.set_ylim(bottom=0)

        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='center left', bbox_to_anchor=(1.2, 0.5))
        fig.tight_layout()
        fig.savefig(self.data[eps]["path"] + "/scores.pdf",
                    bbox_inches='tight',
                    pad_inches=0)

    @resultsLoaded
    def plotCompactness(self, eps, fontSize=8.5, printBest=False):
        font = {'size': fontSize, 'family': 'serif'}
        matplotlib.rc('font', **font)
        matplotlib.rc('text', usetex=True)
        MS = 2
        SYMBOL = "o"
        data = self.getResultsWithMetrics(eps)

        fig = plt.figure()
        ax1 = plt.gca()
        # Size for 10a and b
        #fig.set_size_inches(2.15, 1.6)
        # Size for 7c
        fig.set_size_inches(2.2, 1.7)

        if self.gt_eps is not None:
            data_gt = self.getResultsWithMetrics(self.gt_eps)
            if data_gt is not None:
                plotEps = round(math.log(self.gt_eps, 10)) if eps > 0 else 0
                if plotEps != 0:
                    plotEps = "10^{" + str(plotEps) + "}"
                l3 = ax1.plot(data_gt.perimeter / 1e3,
                              data_gt.area / 1e6,
                              SYMBOL,
                              label=rf"Solutions, $\varepsilon={plotEps}$",
                              markersize=MS,
                              color="black",
                              alpha=0.2)
        #l2 = ax1.plot(data.perimeter / 1e3,
        #              data.area / 1e6,
        #              "--",
        #              linewidth=MS - 7,
        #              label="Conv. hull, approx. solutions",
        #              color="red")
        plotEps = round(math.log(self.data[eps]["eps"], 10)) if eps > 0 else 0
        if plotEps != 0:
            plotEps = "10^{" + str(plotEps) + "}"
        if eps != self.gt_eps:
            ax1.plot(data.perimeter / 1e3,
                     data.area / 1e6,
                     SYMBOL,
                     label=rf"Solutions, $\varepsilon={plotEps}$"
                     if eps > 0 else "Solutions",
                     markersize=MS,
                     alpha=1,
                     color="red" if eps != 0 else "black")

        ax1.set_xlabel("Perimeter [km]")
        ax1.set_ylabel(r"Area [km$^2$]", rotation='horizontal')
        ax1.yaxis.set_label_coords(0.05, 1.02)
        ax1.grid(True)
        ax1.legend(handletextpad=0, borderpad=0.2, handlelength=1)
        fig.tight_layout()

        #plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        fig.savefig(self.data[eps]["path"] + "/hull.pdf",
                    bbox_inches='tight',
                    pad_inches=0)


#-------------------------------------------------------------------------------


def timer(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} took: {(time()-start):.3f}s")
        return result

    return wrapper


@timer
def readSquentialWkb(self, eps):
    table = self.data[eps]["data_table"]
    path = self.data[eps]["path_table"]
    polyPath = path.replace("_results.csv", "_union.p")
    alreadyProcessed = os.path.exists(polyPath)

    #polys = []
    unionPolys = []
    filenames = []
    for numberOfPolygons in sorted(table.numberOfPolygons):
        file = f"/{int(numberOfPolygons):0>8}.bin"
        filename = self.data[eps]["path"] + file
        filenames.append(filename)
        #polys.append(readWkbFile(filename))
    self.data[eps]["filenames"] = filenames

    if alreadyProcessed:
        with open(polyPath, "rb") as fp:
            self.data[eps]["unionPolys"] = pickle.load(fp)
    else:

        def polyGenerator(names):
            lastPoly = None
            for n in names:
                poly = readWkbFile(n)
                if not lastPoly:
                    lastPoly = unary_union(poly)
                else:
                    lastPoly = unary_union([poly, lastPoly])
                yield lastPoly.simplify(1e-6, preserve_topology=False)
                self.all_wkt.append(poly) # Append each poly to the list
                self.wkt_union.append(lastPoly) # Append union of poly and lastPoly
        self.data[eps]["unionPolys"] = list(polyGenerator(filenames))
    #return polys, unionPolys, filenames


def readResults(path):
    table = pd.read_csv(path)
    if "area" in table.columns:
        return table.sort_values("area", ascending=True)
    return table.sort_values(table.columns[0])
