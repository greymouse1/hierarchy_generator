from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection

from shapely import affinity
from shapely.geometry import Polygon, MultiPolygon
import glob
from utils import readWkbAsUnion, readWkbFile
from descartes import PolygonPatch


def drawPoly(poly,
             alpha=1,
             fc=(0.5, 0.5, 0.5),
             ec=None,
             lw=0,
             rotate=0,
             linestyle="solid"):
  patches = []
  if (ec is None):
    ec = fc
  if type(poly) is Polygon:
    if rotate != 0:
      poly = affinity.rotate(poly, rotate)
    patches.append(
        PolygonPatch(poly,
                     fc=fc,
                     ec=ec,
                     alpha=alpha,
                     linewidth=lw,
                     linestyle=linestyle))
  elif type(poly) is MultiPolygon:
    if rotate != 0:
      poly = affinity.rotate(poly, rotate)
    for p in poly.geoms:
      patches.extend(drawPoly(p, alpha, fc, ec, lw, rotate, linestyle))
  return patches


def plotAndSave(filename, withcolor=False):
  if withcolor:
    if "00000001.bin" in filename or "00000000.bin" in filename:
      return

  fig, axs = plt.subplots()
  #fig.set_size_inches(16.80, 10.25)
  axs.set_aspect('equal', 'datalim')
  plt.axis('off')
  fig.tight_layout()
  plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
  plt.margins(0, 0)
  polys = readWkbAsUnion(filename)
  axs.set_xlim(polys.bounds[0::2])
  axs.set_ylim(polys.bounds[1::2])
  patches = []
  if withcolor:
    patches.extend(
        drawPoly(polys,
                 fc=(200 / 256, 200 / 256, 200 / 256),
                 ec=(150 / 256, 0, 0),
                 lw=4))
    if "00000001.bin" in filename:
      buildings = filename[:-12] + "00000001.bin"
      patches.extend(
          drawPoly(readWkbAsUnion(buildings),
                   fc=(130 / 256, 130 / 256, 130 / 256),
                   ec=(0, 0, 0),
                   lw=1.5))
    elif "00000000.bin" in filename:
      triangles = filename[:-12] + "00000000.bin"
      patches.extend(
          drawPoly(readWkbFile(triangles), fc="None", ec=(1, 0, 0), lw=0.5))
  else:
    patches.extend(drawPoly(readWkbFile(filename)))
  collection = PatchCollection(patches, match_original=True)
  axs.add_collection(collection)
  fig.savefig(filename.replace(".bin", ".pdf"),
              bbox_inches='tight',
              pad_inches=0,
              transparent=True)
  plt.close(fig)
  return


def createGif(dataset, files):
  import os
  s_file = files[-1]
  folder = os.path.dirname(s_file)
  output_gif = os.path.join(os.path.dirname(folder), f"{dataset}.gif")

  os.system(
      f"ffmpeg -y -framerate 60 -pattern_type glob -i '{folder}/*.jpg' "\
      f"-loglevel quiet {output_gif}"
  )


if __name__ == '__main__':
  from options import getOpts
  from multiprocessing import Pool
  from tqdm import tqdm
  from functools import partial

  args = getOpts()

  process = partial(plotAndSave, withcolor=False)

  for dataset in tqdm(args.dataset):
    print(f"Starting on dataset {dataset}")
    path = args.path.format(dataset,
                            f"_{args.epsilon:.06f}" if args.epsilon > 0 else "")
    print(path)
    files = glob.glob(path + "/*.bin")
    print(files)
    with Pool() as p:
      p.map(process, files)
    if args.gif:
      createGif(dataset, files)