#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import random

import numpy as np

parser = argparse.ArgumentParser(description='Read kernel bank or file, with indexes in first line')
parser.add_argument('-o', '--output', type=argparse.FileType('w', encoding='UTF-8'), help='Output kernel file path')
parser.add_argument('-d', '--dimension', type=int, nargs='*', help='Dimensions of kernel, 1 or 2 arguments',
                    default=[3])
parser.add_argument('-n', '--nkernels', type=int, help='Number of kernels', default=8)
parser.add_argument('-b', '--bands', type=int, help='Number of bands per kernel', default=1)

args = vars(parser.parse_args())
print(args)
OUTPUT = args["output"]
DIMS = args["dimension"]
NKERNELS = args["nkernels"]
BANDS = args["bands"]

if len(DIMS) == 1:
    n = DIMS[0]
    m = DIMS[0]
else:
    n = DIMS[0]
    m = DIMS[1]

output_filename = "kernel-bank-rand-%dx%d.txt" % (n, m)
if OUTPUT is None:
    OUTPUT = open(file=output_filename, mode='w', encoding='UTF-8')
OUTPUT.write("%d %d %d %d\n" % (BANDS, m, n, NKERNELS))

for i in range(NKERNELS):
    weights = [random.randint(0, 100) for j in range(n*m*BANDS)]
    bias = 0.0
    mean = np.mean(weights)
    weights = weights - mean
    norm = np.linalg.norm(weights)
    weights = weights/norm
    weights.tofile(OUTPUT, " ", '%.2f')
    OUTPUT.write(" %.2f\n" % bias)

OUTPUT.close()
print("Saved kernel to %s" % output_filename)

