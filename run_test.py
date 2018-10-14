#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import os
import re
import subprocess

import numpy as np

parser = argparse.ArgumentParser(description='Read kernel bank or file, with indexes in first line')
parser.add_argument('-s', '--split-dir', type=str, help='splits path', default='split')

args = vars(parser.parse_args())
print(args)

SPLIT = args["split_dir"]

splits = set()
for file in os.listdir(SPLIT):
    path = os.path.join(SPLIT, file)
    if os.path.isfile(path) and file.startswith("train"):
        splits.add((path, re.sub('train', 'test', path)))


def parse(lines):
    for line in lines.split('\n'):
        line = str(line)
        if line.startswith("Hit:"):
            match = re.search(r'Hit: (?P<hit>\d+) Miss: (?P<miss>\d+)', line)
            hit = int(match.group('hit'))
            miss = int(match.group('miss'))
            return hit / (hit + miss)


subprocess.run(["make", "srcs"])

for kernel in os.listdir('.'):
    if os.path.isfile(kernel) and kernel.startswith("kernel-bank"):
        scores = []
        print("Running 5x2 CV on %s" % kernel)
        output_params_dir = "params"
        os.makedirs(output_params_dir, exist_ok=True)
        for train, test in splits:
            print(train, end=' ', flush=True)
            parameter_path = os.path.join(output_params_dir, "%s-%s-output-parameters.txt" % (kernel[:-4], train[-10:-4]))
            subprocess.run(["bin/training", train, kernel, parameter_path], stdout=subprocess.PIPE)
            result = subprocess.run(["bin/testing", test, kernel, parameter_path], stdout=subprocess.PIPE)
            scores.append(parse(result.stdout.decode('utf-8')))

            print(test, end=' ', flush=True)
            parameter_path = os.path.join(output_params_dir, "%s-%s-output-parameters.txt" % (kernel[:-4], test[-9:-4]))
            subprocess.run(["bin/training", test, kernel, parameter_path], stdout=subprocess.PIPE)
            result = subprocess.run(["bin/testing", train, kernel, parameter_path], stdout=subprocess.PIPE)
            scores.append(parse(result.stdout.decode('utf-8')))

        print("")
        scores = np.array(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


