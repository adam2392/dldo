#!/usr/bin/env python3.7

# Copyright 2020, Gurobi Optimization, LLC

# Solve a traveling salesman problem on a randomly generated set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.

import random
from itertools import combinations
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB


def subtourelim(model, where):
    """
    Callback - use lazy constraints to eliminate sub-tours.

    Parameters
    ----------
    model :
    where :

    Returns
    -------

    """
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist((i, j) for i, j in model._vars.keys()
                                if vals[i, j] > 0.5)
        # find the shortest cycle in the selected edge list
        tour = subtour(selected)
        if len(tour) < n:
            # add subtour elimination constr. for every pair of cities in tour
            model.cbLazy(gp.quicksum(model._vars[i, j]
                                     for i, j in combinations(tour, 2))
                         <= len(tour) - 1)


def subtour(edges):
    """
    Given a tuplelist of edges, find the shortest subtour.

    Parameters
    ----------
    edges :

    Returns
    -------

    """
    unvisited = list(range(n))
    cycle = range(n + 1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle


# Parse argument
if __name__ == '__main__':
    n = 10
    # Create n random points

    random.seed(1)

    # read in distances from file
    data_dir = Path(Path("/Users/adam2392/Documents/dldo") / 'data' / 'raw')
    data_fname = 'usa50.txt'
    data_vec = []

    # build distance matrix
    dist = {}
    n = 0
    with open(Path(data_dir / data_fname), 'r') as fin:
        for i, line in enumerate(fin):
            line_as_list = line.split()
            for j, num in enumerate(line_as_list):
                if int(num) != 0:
                    dist[(i, j)] = float(num)
            n += 1
    print(n)

    # initialize a Gurobi model
    m = gp.Model()

    # Create variables
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i, j in vars.keys():
        vars[j, i] = vars[i, j]  # edge in opposite direction

    # Add degree-2 constraint
    m.addConstrs(vars.sum(i, '*') == 2 for i in range(n))

    # Optimize model
    m._vars = vars
    m.Params.lazyConstraints = 1
    m.optimize(subtourelim)

    # get values of subtour
    vals = m.getAttr('x', vars)
    selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

    # get the subtour
    tour = subtour(selected)

    assert len(tour) == n
    print('')
    print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g' % m.objVal)
    print('')
