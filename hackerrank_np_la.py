#!/bin/python3

import numpy as np

# Complete the whichSection function below.
def findDet(a):
    # Return the section number you will be assigned to assuming you are student number k.
#    print('array a: {0}'.format(a))
    print(np.around(np.linalg.det(np.array(a)), 2) # had to round to address "expected format of 2 decimal places"

if __name__ == '__main__':

    t = int(input())
#    assert 1 <= t <= 250
    a = []
    for t_itr in range(t):
        a.append([float(item) for item in input().split()])
    findDet(a)
