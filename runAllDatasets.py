#!/usr/bin/env python3
import sys

sys.path.append("scripts")
from utils import Dataline
import numpy as np
import warnings
import math

warnings.filterwarnings("ignore")


def processLineWrapper(dataset):
    return dataset.getDataLine(1e-6)


if __name__ == '__main__':
    from utils import TempArgs
    import os
    from concurrent.futures import ProcessPoolExecutor as Pool
    import psutil
    from dataset import Dataset
    from matplotlib import pyplot as plt
    #import argparse
    #parser = argparse.ArgumentParser(description='Process some inputs.')
    #parser.add_argument("--threads", "-t", type=int, help="number of threads")
    #args2 = parser.parse_args()

    #set path to the directory that includes the subfolder 'data'
    path_string = "/Users/shark/Desktop/My Documents/uni/Munster/Possible_thesis/Bonn/virtual_folder/pythonProject/testing/"
    #set path including the subfolder data (this is for collecting all datasets)
    path = "/Users/shark/Desktop/My Documents/uni/Munster/Possible_thesis/Bonn/virtual_folder/pythonProject/testing/data"
    method = "tri"
    folder = method
    args = TempArgs(epsilon=0.1, path=folder + "/{}{}")

    datasets = []
    #Retrieve all datasets
    for r, d, f in os.walk(path, topdown=False):
        for dataset in d:
            datasets.append(Dataset(dataset, args.path, args.epsilon))
    #datasets = [f"gruppe{i+1}" for i in range(12)]

    memory = round(psutil.virtual_memory()[0] / 1024**3)

    # Run adopt merge for all datasets
    epsilons = [0.1, 1e-6]
    for dataset in sorted(datasets):
        command = f"java -Xmx{1}G -jar generalization.jar " + \
        f"-m {method} -d {dataset.name} -rf {folder} " + \
        f"-t {1} -p \"{path_string}\""
        print(command)
        os.system(command)
        # estr = ""
        # for epsilon in epsilons:
        #     if not os.path.exists(dataset.data[epsilon]["path_table"]):
        #         if len(estr) > 0:
        #             estr += ","
        #         estr += f"{epsilon}"
        # if len(estr) > 0:
        #     command = f"java -Xmx{memory-8}G -jar generalization.jar " + \
        #     f"-m {method} -d {dataset.name} -e {estr} -rf {folder} " + \
        #     f"-t {args2.threads} -p {path_string}"
        #     print(command)
        #     os.system(command)

    # if True:
    #     with Pool(args2.threads) as p:
    #         lines = p.map(processLineWrapper, sorted(datasets))
    #     lines = list(lines)
    # else:
    #     lines = []
    #     for dataset in sorted(datasets):
    #         lines.append(processLineWrapper(dataset))

    # bestAlpha = []
    # for l in lines:
    #     bestAlpha.append(l.x[np.nanargmax(l.y)])

    print(f"Number of datasets: {len(datasets)}")
