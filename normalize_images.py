#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np

import cv2

parser = argparse.ArgumentParser(description='Read all images on path and normalize then')
parser.add_argument('-i', '--input', type=str, help='Input images path',
                    required=True)
parser.add_argument('-o', '--output', type=str, help='Output images path')

args = vars(parser.parse_args())
print(args)
INPUT = args["input"]
OUTPUT = args["output"]

for file in os.listdir(INPUT):
    if not os.path.isdir(file) and file.endswith(".png"):
        img_orig = cv2.imread(os.path.join(INPUT, file), cv2.IMREAD_ANYCOLOR)
        print("%s  \tmax: %d, min:%d" % (file, np.amax(img_orig), np.amin(img_orig)))
        #print(file + "  \t" + str(img_orig.shape))
        # TODO align histograms
