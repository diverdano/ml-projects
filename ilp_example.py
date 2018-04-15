#-- example from https://pypi.python.org/pypi/PuLP/1.1 --
from pulp import *

prob = LpProblem("test1", LpMinimize)

# Variables
x = LpVariable("x", 0, 4)
y = LpVariable("y", -1, 1)
z = LpVariable("z", 0)

# Objective
prob += x + 4*y + 9*z

# Constraints
prob += x+y <= 5
prob += x+z >= 10
prob += -y+z == 7

#GLPK().solve(prob) # commercial solver library (just use open source solver)
prob.solve()

# Solution
for v in prob.variables():
    print(v.name, "=", v.varValue)
