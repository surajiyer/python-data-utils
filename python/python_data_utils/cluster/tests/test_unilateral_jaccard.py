from python_data_utils.cluster.unilateral_jaccard import ujaccard_similarity_score


def test_1():
    # Test toy example w/ single thread processing and depth=1
    # Toy example directly from the paper
    # edges = [{1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {2, 7}, {3, 7},
    #          {4, 7}, {7, 8}, {7, 9}, {7, 10}, {7, 11}, {8, 12},
    #          {9, 12}, {10, 12}, {11, 12}]
    # nodes = set().union(*edges)
    # nodes = [set().union(*[e for e in edges if n in e]) for n in nodes]
    nodes = list(range(1, 13))
    N = len(nodes)
    edges = {1: {2, 3, 4, 5, 6}, 2: {1, 7}, 3: {1, 7}, 4: {1, 7}, 5: {1}, 6: {1},
             7: {2, 3, 4, 8, 9, 10, 11}, 8: {7, 12}, 9: {7, 12}, 10: {7, 12},
             11: {7, 12}, 12: {8, 9, 10, 11, 12}}
    array = ujaccard_similarity_score((nodes, edges), depth=1, n_jobs=1)
    assert array.shape == (N, N)


def test_2():
    # Test toy example w/ multiprocessing and depth=1
    # Toy example directly from the paper
    nodes = list(range(1, 13))
    N = len(nodes)
    edges = {1: {2, 3, 4, 5, 6}, 2: {1, 7}, 3: {1, 7}, 4: {1, 7}, 5: {1}, 6: {1},
             7: {2, 3, 4, 8, 9, 10, 11}, 8: {7, 12}, 9: {7, 12}, 10: {7, 12},
             11: {7, 12}, 12: {8, 9, 10, 11, 12}}
    array = ujaccard_similarity_score((nodes, edges), depth=1, n_jobs=-1)
    assert array.shape == (N, N)


def test_3():
    # Test toy example w/ single thread processing and depth=2
    nodes = list(range(1, 13))
    N = len(nodes)
    edges = {1: {2, 3, 4, 5, 6}, 2: {1, 7}, 3: {1, 7}, 4: {1, 7}, 5: {1}, 6: {1},
             7: {2, 3, 4, 8, 9, 10, 11}, 8: {7, 12}, 9: {7, 12}, 10: {7, 12},
             11: {7, 12}, 12: {8, 9, 10, 11, 12}}
    array = ujaccard_similarity_score((nodes, edges), depth=2, n_jobs=1)
    assert array.shape == (N, N)


def test_4():
    # Test toy example w/ single thread processing and depth=3
    nodes = list(range(1, 13))
    N = len(nodes)
    edges = {1: {2, 3, 4, 5, 6}, 2: {1, 7}, 3: {1, 7}, 4: {1, 7}, 5: {1}, 6: {1},
             7: {2, 3, 4, 8, 9, 10, 11}, 8: {7, 12}, 9: {7, 12}, 10: {7, 12},
             11: {7, 12}, 12: {8, 9, 10, 11, 12}}

    array = ujaccard_similarity_score((nodes, edges), depth=3, n_jobs=1)
    assert array.shape == (N, N)
