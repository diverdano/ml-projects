#! /usr/bin/env python

# -- imports --

import csv
import numpy as np
from scipy.optimize import minimize # doesn't do integer linear programming
from pulp import *

# -- helper functions --

def readCsvFile(fname):
    with open(fname, 'r') as inf:
        return list(csv.reader(inf))

# -- get data --

items           = readCsvFile('projects/datasmart/calories.csv')
numItems        = len(items)
item_array      = np.array(items[1:numItems])               # start at 1 to drop the header row
item_list       = item_array[...,0]                         # list of items
cal_array       = item_array[...,1].astype(int)             # array of items in second column

# -- setup integer linear program --

# objective:
    # minimize number of concession items needed to acheive 2,400 calories
    # x0 + x1 + ... + x14 = 2400
    # convert to normal form
    # 0 = -2400 + x0 + x1 + ... + x14
# constraints
    # count for each concession must be an integer

prob = LpProblem("test_ch1", LpMinimize)

# Variables
for i, item in enumerate(item_list):
    item = item.replace(' ','')
    globals()['var_{0}'.format(i)] = LpVariable(item, 0, None, LpInteger)

# Objective
prob += var_0 + var_1 + var_2 + var_3 + var_4 + var_5 + var_6 + var_7 + var_8 + var_9 + var_10 + var_11 + var_12 + var_13

# Constraints
prob += 200*var_0 + 0*var_1 + 255*var_2 + 300*var_3 + 300*var_4 + 320*var_5 + 265*var_6 + 240*var_7 \
        + 280*var_8 + 560*var_9 + 480*var_10 + 500*var_11 + 150*var_12 + 120*var_13 == 2400

# -- run ilp --

prob.solve()

# -- output results --

# Solution
for v in prob.variables():
    print("{} = {}".format(v.varValue, v.name))
