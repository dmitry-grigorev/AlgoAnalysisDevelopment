
# start - vertex from which dfs will be initiated
# matrix - adjacency matrix of graph
# visited - boolean array of visited vertices
# path - list of vertices that were visited
def dfs_adj(start, matrix, visited, path=None):
    if path is None:
        path = list()
    visited[start] = True
    path.append(start)
    for i in range(len(matrix[start])):
        if matrix[start][i] == 1 and not visited[i]:
            dfs_adj(i, matrix, visited, path)
    return path


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