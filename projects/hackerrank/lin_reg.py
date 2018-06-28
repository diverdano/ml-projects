#!/bin/env python
from scipy.stats import linregress
import numpy

a = [15, 12, 8, 8, 7, 7, 7, 6, 5, 3]
b = [10, 25, 17, 11, 13, 17, 20, 13, 9, 15]
lr = linregress(a, b)
cc = numpy.corrcoef(a, b)
'{0:.3f}'.format(cc[0, 1])
