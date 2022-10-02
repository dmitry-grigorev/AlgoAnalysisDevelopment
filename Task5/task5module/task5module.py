import numpy as np


def generate_from_nm(n, m):
    initial = np.random.permutation([1] * m + [0] * (n * (n - 1) // 2 - m))
    Madj = np.zeros(shape=(n, n), dtype="int")
    l = n - 1
    start = 0
    end = l
    for i in range(1, n):
        Madj[i - 1, i:] = initial[start:end]
        l = n - i
        start = end
        end += l - 1
    Madj += Madj.T
    return Madj


# start - vertex from which dfs will be initiated
# adj_list - adjacency list of graph
def dfs_adj(start, adj_list):
    # list of visited vertices
    visited = [False] * len(adj_list)

    path = list()
    # Create a stack for DFS
    stack = list()

    # Push the current source node.
    stack.append(start)

    while len(stack):
        # Pop a vertex from stack
        start = stack[-1]
        stack.pop()

        # Stack may contain same vertex twice. So
        # we need to print the popped item only
        # if it is not visited.
        if not visited[start]:
            path.append(start)
            visited[start] = True

        # Get all adjacent vertices of the popped vertex s
        # If an adjacent has not been visited, then push it
        # to the stack.
        for node in adj_list[start]:
            if not visited[node]:
                stack.append(node)
    return path


def find_components(adj_list):
    n = len(adj_list)
    components = [-1] * n
    i, compnum = 0, 0
    while i < n:
        if components[i] == -1:
            comp_verts = dfs_adj(i, adj_list)
            for j in range(len(comp_verts)):
                components[comp_verts[j]] = compnum
            compnum += 1
        i += 1
    return components


# start - vertex from which dfs will be initiated
# matrix - adjacency matrix of graph
# vert_to_find - optional, vertex for which the shortest path should be found
# with BFS we always reach a vertex from given source using the minimum number of edges
def bfs_adj(start, matrix, vert_to_find=None):
    # Visited vector to so that a
    # vertex is not visited more than
    # once Initializing the vector to
    # false as no vertex is visited at
    # the beginning
    visited = [False] * len(matrix)
    q = [start]

    # Set source as visited
    visited[start] = True
    path = list()
    while q:
        vis = q[0]

        q.pop(0)
        # appending current node to path
        path.append(vis)

        # For every adjacent vertex to
        # the current vertex
        for i in range(len(matrix)):
            if (matrix[vis][i] == 1 and
                    (not visited[i])):
                # Push the adjacent node
                # in the queue
                q.append(i)

                # set
                visited[i] = True

                # if we got to vert_to_find, then
                # we found the shortest path from start to vert_to_find
                if vert_to_find == i:
                    path.append(i)
                    return path
    return path


def adj_mat_to_adj_list(matrix):
    res = dict()
    for i in range(len(matrix)):
        res[i] = list()
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1:
                res[i].append(j)
    return res
