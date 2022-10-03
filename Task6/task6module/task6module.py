
# A utility function to find the vertex with
# minimum distance value, from the set of vertices
# not yet included in shortest path tree
def minDistance(matrix, dist, sptSet):
    # Initialize minimum distance for next node
    minimum = 1e7
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
    dist = [1e7] * (len(matrix))
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
