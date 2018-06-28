#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the whichSection function below.
def whichSection(n, k, a):
    # Return the section number you will be assigned to assuming you are student number k.
    print('num/students: {0}, student k: {1}, section array: {2}'.format(n, k, a))
    section = None
    # students left = num students, subtract section count from students to set "students placed", if students placed > k, set 'assigned section + 1 (zero based)'
    students_placed = 0
    # enumerate and iterate through a (sections), test if k in - test if k student
    for num, section_size in enumerate(a):
        students_placed += section_size
        if students_placed >= k:
            section = num + 1
            print('section: {0}'.format(section))
            return section
    print('end loop, section: {0}'.format(section))
    return section

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())
    assert 1 <= t <= 250
    for t_itr in range(t):
        nkm = input().split()
        n = int(nkm[0])
        k = int(nkm[1])
        m = int(nkm[2])
        a = list(map(int, input().rstrip().split()))
        assert 1 <= k <= n <= 500
        assert 1 <= m <= 500
        assert sum(a) == n
        print('num/sections: {0}'.format(m))
        result = whichSection(n, k, a)

        fptr.write(str(result) + '\n')

    fptr.close()
