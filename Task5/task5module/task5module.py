import numpy as np


def generate_from_nm(n, m):
    """
    :param n: order of graph (# of nodes)
    :param m: size of graph (# of edges)
    :return Madj: random adjacency matrix
    """
    # generate an array of size n*(n-1)/2
    initial = np.random.permutation([1] * m + [0] * (n * (n - 1) // 2 - m))
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


def dfs_adj(start, adj_list):
    """
    :param start: vertex from which dfs will be initiated
    :param adj_list: adjacency list of graph
    :return path: contain all visited vertices of component 'start' belongs to
    """
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

        # Prevent the current vertex from being push twice into the stack
        if not visited[start]:
            path.append(start)
            visited[start] = True

        # push all non-visited neighbors into the stack
        for node in adj_list[start]:
            if not visited[node]:
                stack.append(node)
    return path


def find_components(adj_list):
    """
    :param adj_list: adjacency list of graph
    :return components: contain numbers of components to which each vertex belongs to
    """
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


def bfs_adj(start, adj_list, vert_to_find):
    """
    :param start: vertex from which bfs will be initiated
    :param adj_list: adjacency list of graph
    :param vert_to_find: vertex to which we look for the shortest path
    :return path: one of the shortest paths from 'vert_to_find' to 'start'
    """

    if start == vert_to_find:
        return [start]

    visited = [False] * len(adj_list)  # list of statuses
    predecessors = [start] * len(adj_list)  # list of the nodes' bfs-predecessors
    visited[start] = True
    q = [start]  # queue

    while q:
        vis = q[0]

        q.pop(0)

        for node in adj_list[vis]:
            if node == vert_to_find:  # bfs met searched node
                # then stop the algo
                path = [node]
                prev = vis
                while prev != start:  # backtracking
                    path.append(prev)
                    prev = predecessors[prev]
                path.append(start)
                return path
            if not visited[node]:
                q.append(node)
                visited[node] = True
                predecessors[node] = vis
        visited[vis] = True
    # if queue became empty, then there are no paths
    print("No paths were found")
    return None


def adj_mat_to_adj_list(matrix):
    """
    :param matrix: adjacency matrix of graph
    :return adj_list: adjacency list of the graph
    """
    adj_list = dict()
    for i in range(len(matrix)):
        adj_list[i] = list()
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1:
                adj_list[i].append(j)
    return adj_list
