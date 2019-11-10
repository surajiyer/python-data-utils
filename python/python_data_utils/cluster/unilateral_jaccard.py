# coding: utf-8

"""
    description: Unilateral Jaccard Similarity Coefficient
        https://pdfs.semanticscholar.org/3031/c9f0c265571846cc2bfd7d8ca3538918a355.pdf
    author: Suraj Iyer
"""

__all__ = ['ujaccard_similarity_score', 'calculate_edges_list']

import numpy as np
from collections import deque
import time
import itertools as it
import multiprocessing as mp
from typing import Iterable, Tuple, Mapping, Hashable


def dfs_paths_recursive(
        graph: Mapping, start, goal, depth: int,
        visited: set = set(), n_paths: int = 0) -> int:
    """
    Depth-first search until given depth for finding all paths
    between start and goal vertex on given graph.

    Recursive version.

    :param graph: dict
        Adjacency list of graph.
    :param start:
        key in graph corresponding to source vertex.
    :param goal:
        key in graph corresponding to destination vertex.
    :param depth: int
        distance from start vertex within which to search.
    :param visited: set
        Set of visited nodes. Should be empty on first call.
    :param n_paths: int
        Number of paths from start to goal found so far.
        Should be empty on first call.
    :return: int
        Number of unique paths between start and goal vertex.
    """
    assert depth > -1
    assert n_paths > -1

    # Mark the current node as visited
    visited |= {start}

    # If current vertex is same as destination,
    # then increment n_paths
    if start == goal:
        n_paths += 1
    else:
        # If current vertex is not destination
        # Recur for all the vertices adjacent to this vertex
        if depth > 1:
            for neighbour in graph[start]:
                if neighbour not in visited:
                    n_paths = dfs_paths_recursive(
                        graph, neighbour, goal, depth - 1, visited, n_paths)
        else:
            # if there is room for only one more element on the path,
            # check directly if goal vertex is a neighbor. If not, no
            # need to go through all the vertex's neighbors anymore.
            if goal in graph[start]:
                n_paths += 1

    # Mark current vertex as unvisited
    visited.remove(start)
    return n_paths


def dfs_paths(graph: Mapping, start, goal, depth: int) -> list:
    """
    Depth-first search until given depth for finding all paths
    between start and goal vertex on given graph.

    Iterative version.

    :param graph: dict
        Adjacency list of graph.
    :param start:
        key in graph corresponding to source vertex.
    :param goal:
        key in graph corresponding to destination vertex.
    :param depth: int
        distance from start vertex within which to search.
    :return: list
        All unique paths between start and goal vertex.
    """
    assert depth > -1

    stack = deque([(start, {start})])
    n_paths = 0

    # edge case: start == goal
    if start == goal:
        n_paths += 1

    while stack:
        (vertex, visited) = stack.popleft()
        if goal in visited:
            n_paths += 1
        else:
            if len(visited) < depth:
                stack.extend([(neighbour, visited | {neighbour})
                              for neighbour in graph[vertex]
                              if neighbour not in visited])
            else:
                # if there is room for only one more element on the path,
                # check directly if goal vertex is a neighbor. If not, no
                # need to go through all the vertex's neighbors anymore.
                if goal in graph[vertex]:
                    n_paths += 1

    return n_paths


def calculate_edges_list(documents: Iterable[set]) -> dict:
    """
    Compare every document with each other and check for intersection
    in words. If doc_{i} ∩ doc_{j}, then create an edge between i and
    j.

    :param documents: Iterable of sets, [set,...]
        The sets represent a single document / item. The elements of
        the set are essentially the "words".
    :return: dict
        An adjacency mapping where key = document index i and
        value = set of document indices connected to documents[i].
    """
    # compute adjacency list of edges between documents
    r = range(len(documents))
    edges: dict = {i: set() for i in r}
    start = time.time()

    def is_edge(args):
        i, j = args
        if len(documents[i] & documents[j]) > 0:
            edges[i] |= {j}
            edges[j] |= {i}

    deque(map(is_edge, it.combinations(r, 2)), maxlen=0)
    print(f"calculating edges, Time to run: {time.time() - start}")
    return edges


def calculate_edges_matrix(documents: Iterable[set]) -> np.ndarray:
    """
    Compare every document with each other and check for intersection
    in words. If doc_{i} ∩ doc_{j}, then create an edge between i and
    j.

    :param documents: Iterable of sets [set,...]
        The sets represent a single document / item. The elements of
        the set are essentially the "words".
    :return: np.ndarray
        An adjacency matrix.
    """
    documents = np.array(documents)
    # return np.vectorize(lambda x: int(len(x) > 0))(
    #     documents[None, :] & documents[:, None])
    return np.vectorize(lambda x: int(len(x) > 0))(
        np.bitwise_and.outer(documents, documents))


def _consumer(
        q: mp.Queue, G: Tuple[Iterable[set], Mapping[Hashable, set]],
        depth: int, return_dict):
    """
    Consumer process for the ujaccard_similarity_score function.

    :param q: Multiprocessing queue
    :param G: (V, E)
        V = Iterable of sets [set,...]
            Each set represent a single document / item. The elements of
            the set are essentially the "words".
        E = Mapping of sets
            The key represents the index of a single document in V and
            the value is a set of indices corresponding to documents it
            is linked with.
    :param depth: int
        Max. depth up to which to calculate ujaccard score.
    :param return_dict
        Dictionary shared between processes to store final result.
    """
    assert depth > -1
    V, E = G

    while True:
        # get arguments from queue
        args = q.get()
        if args is None:
            break
        i, j = args

        # #_paths from i to j equals # paths from j to i
        # start = time.time()
        # n_paths = dfs_paths(edges, i, j, depth)
        # print(f"dfs_paths\t\tDoc{i}, Doc{j}, # Paths: {n_paths}, \
        #     Time to run: {time.time() - start}")
        # start = time.time()
        n_paths = dfs_paths_recursive(E, V[i], V[j], depth)
        # print(f"dfs_paths_recursive\tDoc{i}, Doc{j}, # Paths: {n_paths}, \
        #     Time to run: {time.time() - start}")
        return_dict[(i, j)] = n_paths


def ujaccard_similarity_score(
        G: Tuple[Iterable[set], Mapping[Hashable, set]],
        depth: int = 1, n_jobs: int = -1) -> np.ndarray:
    """
    Given a graph G where nodes = documents and edges
    represents intersection between documents. The uJaccard
    similarity score is the number of paths in G between pairs
    of documents within maximum :depth: distance divided by
    total number of outgoing paths.

    uJaccard(u, v) = |paths(u, v, depth)| / |edges(u)|

    :param G: (V, E)
        V = Iterable of sets [set,...]
            Each set represent a single document / item. The elements of
            the set are essentially the "words".
        E = Mapping of sets
            The key represents the index of a single document in V and
            the value is a set of indices corresponding to documents it
            is linked with.
    :param depth: int
        Maximum path length between documents.
    :param n_jobs: int, Optional (default=None)
        if None, use all cpu resources, else only the number specified.
    :return: np.ndarray
        Similarity matrix \forall v \in V.
    """
    assert isinstance(depth, int) and depth > 0
    assert isinstance(n_jobs, int) and n_jobs >= -1

    # Pre-compute the number of outgoing edges from document i
    V, E = G
    n_edges = np.array([len(E[i]) for i in V])
    mask = n_edges > 0

    # Initialize the similarity matrix
    uJaccard = np.identity(len(V), dtype=float)

    # initialize multiprocessing queue
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    q = mp.Queue(maxsize=2 * n_jobs + 1)

    # initialize dictionary for return values shared between
    # producer and consumer processes
    return_dict = mp.Manager().dict()

    # initialize the worker processes
    pool = mp.Pool(n_jobs, initializer=_consumer,
                   initargs=(q, G, depth, return_dict))

    # produce items into queue
    for i, j in it.combinations(np.nonzero(mask)[0], 2):
        q.put((i, j))

    # tell workers we're done
    for i in range(n_jobs):
        q.put(None)

    # workers will now properly terminate when all
    # work already assigned has completed
    pool.close()

    # wait for workers to finish
    pool.join()

    # write output to similarity matrix
    for (i, j), value in return_dict.items():
        uJaccard[(i, j), (j, i)] = value
    uJaccard[:, mask] /= n_edges[mask]

    return uJaccard
