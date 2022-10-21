import numpy as np
from typing import List
import copy
import timeit
import matplotlib.pyplot as plt
from Task6.task6module.task6module import generate_from_nm_weighted


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


def test_time_complexity_APSP(Madj: np.ndarray, n):
    exectime = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        subMadj = Madj[:i, :i].tolist()
        start = timeit.default_timer()
        improved_all_pairs_shortest_paths(subMadj)
        exectime[i - 1] = (timeit.default_timer() - start) * 1000
    return exectime


def plot_exectime(times: np.ndarray, fitfunc, strfitfunc: str):
    n = times.shape[0]
    n_seq = np.linspace(1, n, n)
    fit = np.polyfit(fitfunc(n_seq), times, 1)
    print(fit[0])

    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(x=n_seq, y=times, c="b")
    plt.plot(n_seq, fit[1] + fit[0] * fitfunc(n_seq), label="$y = {}$".format(round(fit[0], 5) * 1e5) + \
                                                            "$\cdot 10^{-5}$" + strfitfunc, c="r")
    plt.xlabel("Number of nodes")
    plt.ylabel("Execution time (in milliseconds)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.xticks(rotation = 45)
    ax.grid(True)

    return None


def plot_algo_complexity_info(maxn: int, fitfunc: callable, strfitfunc: str):
    times = np.zeros(maxn, dtype=float)
    weights = [10, 20, 30, 50, 100, 300]
    for _ in range(5):
        times += test_time_complexity_APSP(generate_from_n_full_weighted_directed_graph(maxn, weights), maxn)
    return plot_exectime(times / 5, fitfunc, strfitfunc)


class Graph:

    def __init__(self, Madj: List[list]):
        self.V = len(Madj)  # No. of vertices
        self.graph = []
        for i in range(self.V):
            for j in range(i, self.V):
                if i != j and Madj[i][j] != 0.:
                    self.graph.append([i, j, Madj[i][j]])
        # to store graph

    # function to add an edge to graph
    def addEdge(self, u: int, v: int, w):
        self.graph.append([u, v, w])

    # A utility function to find set of an element i
    # (truly uses path compression technique)
    def find(self, parent: list, i: int):
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

        self.graph = sorted(self.graph, key=lambda item: item[2])

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


def test_time_complexity_Kruskal(Madj: np.ndarray, n):
    exectime = np.zeros(n, dtype=float)
    g = Graph(Madj=Madj.tolist())
    subGraph = Graph([[]])
    for i in range(1, n + 1):
        subGraph.graph.append(g.graph[i-1])
        start = timeit.default_timer()
        subGraph.KruskalMST()
        exectime[i - 1] = (timeit.default_timer() - start) * 1000
    return exectime


def plot_Kruskal_complexity_info(maxn: int, fitfunc: callable, strfitfunc: str):
    times = np.zeros(maxn*(maxn-1)//2, dtype=float)
    weights = [10, 20, 30, 50, 100, 300]
    for _ in range(5):
        times += test_time_complexity_Kruskal(generate_from_nm_weighted(maxn, maxn*(maxn-1)//2, weights), maxn*(maxn-1)//2)
    return plot_exectime(times / 5, fitfunc, strfitfunc)
