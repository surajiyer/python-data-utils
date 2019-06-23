# coding: utf-8

"""
    description: Unilateral Jaccard Similarity Coefficient
        https://pdfs.semanticscholar.org/3031/c9f0c265571846cc2bfd7d8ca3538918a355.pdf
    author: Suraj Iyer
"""

import numpy as np


def _get_all_paths_util(u, v, edges, depth, all_paths, visited, path):
    # If exceeds depth, return without doing anything
    if depth == 0:
        return

    # Mark the current node as visited and store in path
    visited[u] = True
    path.append(u)

    # If current vertex is same as destination, then print
    # append the path to all paths list
    if u == v:
        all_paths.append(path[:])
    else:
        # If current vertex is not destination
        # Recur for all the vertices adjacent to this vertex
        for i, e in enumerate(edges[u]):
            if e == 1 and not visited[i]:
                _get_all_paths_util(i, v, edges, depth - 1,
                                    all_paths, visited, path)

    # Remove current vertex from path[] and mark it as unvisited
    path.pop()
    visited[u] = False


def _n_paths_u_to_v(u, v, edges, depth):
    assert depth > 0
    visited = [False] * len(edges[0])
    paths_to_v = []
    _get_all_paths_util(u, v, edges, depth, paths_to_v, visited, [])
    return len(paths_to_v)


def unilateral_jaccard(documents: list, depth: int=1) -> np.ndarray:
    """
    Compute a graph G where nodes = documents and edges
    represents intersection in words occurring between documents.
    The uJaccard similarity score is the number of paths in G
    between pairs of documents within maximum :depth: distance.

    uJaccard(A, B) = |paths(A, B, depth)| / |edges(A)|

    :param documents:  list of sets [set,...]
        The sets represent a single document / item. The elements of
        the set are essentially the "words".
    :param depth: int
        Maximum path length between documents.
    """
    document_edges = [int(len(doc1.intersection(doc2)) > 0)
                      if i != i + j else 0 for i, doc1 in enumerate(documents)
                      for j, doc2 in enumerate(documents[i:])]
    from ..numpy_utils import create_symmetric_matrix
    document_edges = create_symmetric_matrix(document_edges)
    uJaccard = np.full((len(documents),) * 2, -1.)

    for i, j in np.ndindex(*uJaccard.shape):
        # Similarity score between same nodes is always 100%
        if i == j:
            uJaccard[i, j] = 1.
            continue

        # Compute the number of outgoing edges from node i
        n_edges = sum(document_edges[i])

        # Check > 0 to avoid div by 0.
        if n_edges > 0:
            # # paths from i to j equals # paths from j to i, therefore,
            # compute # paths from i to j if # paths from j to i is not
            # already computed else use that
            x = _n_paths_u_to_v(
                i, j, document_edges, depth) if uJaccard[j, i] == -1. \
                else uJaccard[j, i]
            uJaccard[i, j] = x / n_edges
        else:
            uJaccard[i, j] = 0.

    return uJaccard
