#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

import numpy as np

parser = argparse.ArgumentParser(description='Read kernel bank or file, with indexes in first line')
parser.add_argument('-i', '--input', type=argparse.FileType('r', encoding='UTF-8'), help='Input image path',
                    required=True)
parser.add_argument('-o', '--output', type=argparse.FileType('w', encoding='UTF-8'), help='Output image path')

args = vars(parser.parse_args())
print(args)
INPUT = args["input"]
OUTPUT = args["output"]
if OUTPUT is None:
    OUTPUT = open(file=INPUT.name[:-4] + "-norm.txt", mode='w', encoding='UTF-8')
header = INPUT.readline()
OUTPUT.write(header)

for line in INPUT:
    numbers = list(map(float, line.split()))
    bias = numbers[-1]
    weights = numbers[:-1]
    mean = np.mean(weights)
    weights = weights - mean
    norm = np.linalg.norm(weights)
    weights = weights/norm
    weights.tofile(OUTPUT, " ")
    OUTPUT.write(" %s\n" % bias)

