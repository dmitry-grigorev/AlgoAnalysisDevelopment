import numpy as np
from typing import List
import copy


def generate_from_n_full_weighted_directed_graph(n, possibleweights=None):
    """
    Generate random adjacency matrix for simple full directed weighted graph
    :param n: order of graph (# of nodes)s
    :param possibleweights: list of possible nonzero weights of graph's edges
    :return Madj: random adjacency matrix for simple full directed weighted graph
    """
    if possibleweights is None:
        possibleweights = [1]
    # generate an array of size n*(n-1)/2
    initial = np.random.permutation(np.random.choice(possibleweights, size=n * (n - 1) // 2))
    Madj = np.zeros(shape=(n, n), dtype="float")
    l = n - 1
    start = 0
    end = l
    # then this array is transformed into matrix with upper triangle with its elements
    for i in range(1, n):
        Madj[i - 1, i:] = initial[start:end]
        l = n - i
        start = end
        end += l - 1
    # then we define direction of edges
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            if np.random.random(1) > 0.5:
                Madj[j, i] = Madj[i, j]
                Madj[i, j] = 0
            else:
                Madj[j, i] = 0

    return Madj


def specialmul(A: List[list], B: List[list]) -> List[list]:
    n = len(A)
    R = np.zeros(shape=(n, n)).tolist()
    for i in range(n):
        for j in range(n):
            R[i][j] = float("Inf") if i != j else 0
            for k in range(n):
                if R[i][j] > A[i][k] + B[k][j]:
                    R[i][j] = A[i][k] + B[k][j]
    return R


def specialsquare(A: List[list]) -> List[list]:
    return specialmul(A, A)


def improved_all_pairs_shortest_paths(Madj: List[list]) -> List[list]:
    n = len(Madj)
    L = copy.deepcopy(Madj)
    m = 1
    while m < n - 1:
        L = specialsquare(L)
        m *= 2
    return L
