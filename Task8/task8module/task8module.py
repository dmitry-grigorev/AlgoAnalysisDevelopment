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


class Graph:

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []
        # to store graph

    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    # A utility function to find set of an element i
    # (truly uses path compression technique)
    def find(self, parent, i):
        if parent[i] != i:
            # Reassignment of node's parent to root node as
            # path compression requires
            parent[i] = self.find(parent, parent[i])
        return parent[i]

    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x

        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[y] = x
            rank[x] += 1

    # The main function to construct MST using Kruskal's
    # algorithm
    def KruskalMST(self):

        result = []  # This will store the resultant MST

        # An index variable, used for sorted edges
        i = 0

        # An index variable, used for result[]
        e = 0

        # Step 1:  Sort all the edges in
        # non-decreasing order of their
        # weight.  If we are not allowed to change the
        # given graph, we can create a copy of graph

        # structure of edge is [from, to, weight], so we are sorting by 2nd param
        self.graph = sorted(self.graph,
                            key=lambda item: item[2])

        parent = []
        rank = []

        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        # Number of edges to be taken is equal to V-1
        while e < self.V - 1:

            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            # If including this edge doesn't
            # cause cycle, then include it in result
            # and increment the index of result
            # for next edge
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)

        minimum_cost = 0
        for u, v, weight in result:
            minimum_cost += weight
        return result, minimum_cost
