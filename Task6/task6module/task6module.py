import numpy as np
from heapq import heappush, heappop
import matplotlib.pyplot as plt


def generate_from_nm_weighted(n, m, possibleweights=None):
    """
    Generate random adjacency matrix for simple undirected weighted
    :param n: order of graph (# of nodes)
    :param m: size of graph (# of edges)
    :param possibleweights: list of possible nonzero weights of graph's edges
    :return Madj: random adjacency matrix
    """
    if possibleweights is None:
        possibleweights = [1]
    # generate an array of size n*(n-1)/2
    initial = np.random.permutation(np.concatenate((
        np.random.choice(possibleweights, size=m), np.zeros(shape=(n * (n - 1) // 2 - m)))))
    Madj = np.zeros(shape=(n, n), dtype="int")
    l = n - 1
    start = 0
    end = l
    # then this array is transformed into matrix with upper triangle with its elements
    for i in range(1, n):
        Madj[i - 1, i:] = initial[start:end]
        l = n - i
        start = end
        end += l - 1
    # then we reflect upper triangle to lower one
    Madj += Madj.T
    return Madj


# A utility function to find the vertex with
# minimum distance value, from the set of vertices
# not yet included in shortest path tree
def minDistance(matrix, dist, sptSet):
    # Initialize minimum distance for next node
    minimum = float("Inf")
    min_index = -1
    # Search not nearest vertex not in the
    # shortest path tree
    for v in range(len(matrix)):
        if dist[v] < minimum and not sptSet[v]:
            minimum = dist[v]
            min_index = v

    return min_index


# Function that implements Dijkstra's single source
# the shortest path algorithm for a graph represented
# using adjacency matrix representation
def dijkstra(matrix, src):
    dist = [float("Inf")] * (len(matrix))
    dist[src] = 0
    sptSet = [False] * (len(matrix))

    for i in range((len(matrix))):

        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # u is always equal to src in first iteration
        u = minDistance(matrix, dist, sptSet)

        # Put the minimum distance vertex in the
        # shortest path tree
        sptSet[u] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shortest path tree
        for v in range(len(matrix)):
            if (matrix[u][v] > 0 and
                    not sptSet[v] and
                    dist[v] > dist[u] + matrix[u][v]):
                dist[v] = dist[u] + matrix[u][v]

    return dist


def bellman_ford(matrix, src):
    # Step 1: Initialize distances from src to all other vertices
    # as INFINITE
    dist = [float("Inf")] * len(matrix)
    dist[src] = 0

    # Step 2: Relax all edges |V| - 1 times. A simple shortest
    # path from src to any other vertex can have at-most |V| - 1
    # edges
    for _ in range(len(matrix) - 1):
        # Update dist value and parent index of the adjacent vertices of
        # the picked vertex. Consider only those vertices which are still in
        # queue
        for i in range(len(matrix)):
            j = 0
            for w in matrix[i]:
                if w != 0 and dist[i] != float("Inf") and dist[i] + w < dist[j]:
                    dist[j] = dist[i] + w
                j += 1

    # Step 3: check for negative-weight cycles. The above step
    # guarantees the shortest distances if graph doesn't contain
    # negative weight cycle. If we get a shorter path, then there
    # is a cycle.

    for i in range(len(matrix)):
        j = 0
        for w in matrix[i]:
            if w != 0 and dist[i] != float("Inf") and dist[i] + w < dist[j]:
                print("Graph contains negative weight cycle")
                return []
            j += 1

    # return all distances
    return dist


def generate_maze(length, width, numobstacles):
    numcells = length * width
    return np.random.permutation(['X'] * numobstacles + [' '] * (numcells - numobstacles)).reshape(
        (width, length)).tolist()


def neighbors(maze, node):
    width = len(maze)
    length = len(maze[0])
    neigh = list()
    if node[0] > 0:
        neigh.append((node[0] - 1, node[1]))
    if node[0] < width - 1:
        neigh.append((node[0] + 1, node[1]))
    if node[1] > 0:
        neigh.append((node[0], node[1] - 1))
    if node[1] < length - 1:
        neigh.append((node[0], node[1] + 1))
    return neigh


def astar(maze, start, end, heuristics):
    visited = set()
    Q = []
    inQ = set()
    f, g = dict(), dict()
    predecessors = dict()

    g[start] = 0
    f[start] = heuristics(start, end)
    heappush(Q, (f[start], start))
    inQ.add(start)
    while Q:
        # print("im here")
        curr = heappop(Q)
        inQ.remove(curr[1])
        # print((Q, inQ))
        if curr[1] == end:
            path = [end]
            prev = predecessors[curr[1]]
            while prev != start:
                path.append(prev)
                prev = predecessors[prev]
            path.append(start)
            return path
        visited.update({curr[1]: curr[0]})
        for node in neighbors(maze, curr[1]):
            if maze[node[0]][node[1]] != "X":
                score = g[curr[1]] + 1
                if node not in visited or score < g[node]:
                    predecessors.update({node: curr[1]})
                    g[node] = score
                    f[node] = score + heuristics(node, end)
                    if node not in inQ:
                        heappush(Q, (f[node], node))
                        inQ.add(node)

    return "fail"


def plot_maze(maze, start=None, end=None):
    X = np.zeros((10, 20, 3))
    for i in range(10):
        for j in range(20):
            if maze[i][j] == "X":
                X[i, j, 0] = 128
            elif maze[i][j] == "O":
                X[i, j, 1] = 128
            else:
                X[i, j, :] = 255
    if start is not None:
        X[start[0], start[1], 1] = 255
        X[start[0], start[1], 2] = 255
    if end is not None:
        X[end[0], end[1], 0] = 255
        X[end[0], end[1], 1] = 255
        #X[end[0], end[1], 2] = 2
    fig, ax = plt.subplots(facecolor="white")
    ax.imshow(X, interpolation='nearest')
    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if 0 <= col < 20 and 0 <= row < 10:
            z = X[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord
    plt.xticks(np.arange(0, 20, step=1))
    plt.yticks(np.arange(0, 10, step=1))
    plt.show()
