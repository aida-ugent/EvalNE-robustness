#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:
# Contact:
# Date: 22/04/2022

# This file provides implementations of edge attack strategies that target an input graph.
# Both edge removal and addition attacks are supported.


from __future__ import division
import heapq
import numpy as np
import networkx as nx
from scipy.sparse import triu
from evalne.utils import split_train_test as stt


def _top_nonmst_edges(graph_, k):
    # Compute the mST edges of the graph
    mst = nx.tree.minimum_spanning_edges(graph_, algorithm="kruskal", data=False)

    # Remove from the graph the mST edges
    graph_.remove_edges_from(mst)
    edges = list(graph_.edges(data=True))

    if len(edges) < k:
        raise ValueError('Not enough candidate edges outside of the mST!')

    # Select edges with highest weight from the remaining ones
    top_edges = heapq.nlargest(k, edges, key=lambda x: x[2]['weight'])

    return np.array(top_edges)[:, :2].astype(int)


def _top_nonmst_filtr_edges(graph_, k, filter):
    # Compute the mST edges of the graph
    mst = nx.tree.minimum_spanning_edges(graph_, algorithm="kruskal", data=False)

    # Remove from the graph the mST edges
    graph_.remove_edges_from(mst)

    # Remove edges with weight <= filter
    edges = np.array([(u, v) for u, v, attrs in graph_.edges(data=True) if attrs["weight"] > filter])

    if len(edges) < k:
        raise ValueError('Not enough candidate edges!')

    # From the edges with correct weight randomly select k
    idx = np.random.choice(len(edges), k, replace=False)
    return edges[idx]


def _get_edges_ass(graph, coef, make_positive):
    # Get the node degrees and num edges
    deg = np.array(dict(graph.degree()).values())
    m = 2 * len(graph.edges)
    # Compute mean and std over the degrees of each node acting as a source/destination of an edge
    mu = np.sum(np.power(deg, 2)) / m
    std = np.sqrt(np.sum(deg * np.power(deg - mu, 2)) / m)
    # Compute the contribution of each edge to the (dis)assortativity coefficient
    add_val = 0
    if make_positive:
        # Get smallest possible edge contribution
        add_val = np.abs(((np.max(deg) - mu)/std) * ((-mu)/std))
    if coef == 'da':
        res = {(u, v): (((graph.degree(u) - mu) / std) * ((graph.degree(v) - mu) / std)) + add_val
               for u, v in graph.edges}
    elif coef == 'dd':
        res = {(u, v): - ((((graph.degree(u) - mu) / std) * ((graph.degree(v) - mu) / std)) + add_val)
               for u, v in graph.edges}
    else:
        raise ValueError('Unknown coefficient value, options are `da` and `dd`.')
    return res


# ---------------------
# Edge deletion attacks
# ---------------------


def del_edges_rand(graph, k=1):
    edges = list(graph.edges)
    idx = np.random.choice(len(edges), k, replace=False)
    rnd_edges = [edges[i] for i in idx]
    return rnd_edges


def del_edges_rand_nd(graph, k=1):
    # Compute random spanning tree
    E = set(graph.edges)
    st_E = stt.wilson_alg(graph, E)
    other_E = list(E - st_E)
    # Select random edges that are not part of the st
    idx = np.random.choice(len(other_E), k, replace=False)
    rnd_edges = [other_E[i] for i in idx]
    return rnd_edges


def del_edges_ib(graph, k=1, approx=np.inf):
    """ Returns k edges with the highest initial betweenness sorted decreasingly. """
    centrality = nx.edge_betweenness_centrality(graph, k=min(len(graph), approx))
    edges = heapq.nlargest(k, centrality, key=centrality.get)
    return edges


def del_edges_ib_nd(graph, k=1, approx=np.inf):
    """ Returns k edges with the highest initial betweenness sorted decreasingly which do not result in disconnected
    components. """
    graph_ = graph.copy()
    bc = nx.edge_betweenness_centrality(graph_, k=min(len(graph_), approx))
    nx.set_edge_attributes(graph_, values=bc, name='weight')
    return _top_nonmst_edges(graph_, k)


def del_edges_deg(graph, k=1):
    """ Attack edges based on incident node degrees. """
    # line_graph = nx.line_graph(graph)
    # centrality = dict(line_graph.degree())
    dc = {(u, v): (graph.degree(u) + graph.degree(v)) - 2 for u, v in graph.edges}
    nodes = heapq.nlargest(k, dc, key=dc.get)
    return np.array(nodes)


def del_edges_deg_nd(graph, k=1):
    """ Attack edges based on incident node degrees scores without disconnecting the network. """
    # line_graph = nx.line_graph(graph)
    # dc = dict(line_graph.degree())
    dc = {(u, v): (graph.degree(u) + graph.degree(v)) - 2 for u, v in graph.edges}
    graph_ = graph.copy()
    nx.set_edge_attributes(graph_, values=dc, name='weight')
    return _top_nonmst_edges(graph_, k)


def del_edges_pa(graph, k=1):
    """ Attack edges based on preferential attachment scores. """
    pa = {(u, v): graph.degree(u) * graph.degree(v) for u, v in graph.edges}
    edges = heapq.nlargest(k, pa, key=pa.get)
    return np.array(edges)


def del_edges_pa_nd(graph, k=1):
    """ Attack edges based on preferential attachment scores without disconnecting the network. """
    pa = {(u, v): graph.degree(u) * graph.degree(v) for u, v in graph.edges}
    graph_ = graph.copy()
    nx.set_edge_attributes(graph_, values=pa, name='weight')
    return _top_nonmst_edges(graph_, k)


def del_edges_da(graph, k=1):
    """ Attack edges based on degree assortativity. Only for undirected graphs.

    Notes
    -----
    We use the definition of assortativity from [1].
    [1] M. E. J. Newman, Mixing patterns in networks, Physical Review E, 67 026126, 2003.
    """
    graph_ = graph.copy()
    graph_.remove_edges_from(graph_.selfloop_edges())
    da = _get_edges_ass(graph_, 'da', False)
    # Select and return the top k edges
    return heapq.nlargest(k, da, key=da.get)


def del_edges_da_nd(graph, k=1):
    """ Attack edges based on degree assortativity without disconnecting the network. Only for undirected graphs."""
    graph_ = graph.copy()
    graph_.remove_edges_from(graph_.selfloop_edges())
    da = _get_edges_ass(graph_, 'da', True)
    nx.set_edge_attributes(graph_, values=da, name='weight')
    return _top_nonmst_edges(graph_, k)


def del_edges_dd(graph, k=1):
    """ Attack edges based on degree disassortativity. Only for undirected graphs. """
    graph_ = graph.copy()
    graph_.remove_edges_from(graph_.selfloop_edges())
    dd = _get_edges_ass(graph_, 'dd', False)
    return heapq.nlargest(k, dd, key=dd.get)


def del_edges_dd_nd(graph, k=1):
    """ Attack edges based on degree disassortativity without disconnecting the network. Only for undirected graphs."""
    graph_ = graph.copy()
    graph_.remove_edges_from(graph_.selfloop_edges())
    dd = _get_edges_ass(graph_, 'dd', True)
    nx.set_edge_attributes(graph_, values=dd, name='weight')
    return _top_nonmst_edges(graph_, k)


def del_edges_di(graph, node_labels, k=1):
    """ Disconnect internally strategy. Only nodes with the same label are selected as candidates to remove. """
    nls = dict(node_labels)
    candts = []
    for edge in graph.edges():
        if nls[edge[0]] == nls[edge[1]]:
            candts.append(edge)
    # Check that we have enough candidates to return
    if len(candts) < k:
        raise ValueError('Not enough candidate edges with nodes of the same label!')
    indxs = np.random.choice(len(candts), k, replace=False)
    candts = np.array(candts)
    return candts[indxs]


def del_edges_di_nd(graph, node_labels, k=1):
    """ Disconnect internally strategy. Only nodes with the same label are selected as candidates to remove.
     This strategy avoids breaking network connectivity. """
    # Set weight of 2 for edges with same label and 1 for the remaining ones
    nld = dict(node_labels)
    di = {(u, v): 2 if nld[u] == nld[v] else 1 for u, v in graph.edges}
    graph_ = graph.copy()
    nx.set_edge_attributes(graph_, values=di, name='weight')
    return _top_nonmst_filtr_edges(graph_, k, filter=1)


def del_edges_de(graph, node_labels, k=1):
    """ Disconnect externally strategy. Only nodes with different label are selected as candidates to remove. """
    nls = dict(node_labels)
    candts = []
    for edge in graph.edges():
        if nls[edge[0]] != nls[edge[1]]:
            candts.append(edge)
    # Check that we have enough candidates to return
    if len(candts) < k:
        raise ValueError('Not enough candidate edges with nodes of the same label!')
    indxs = np.random.choice(len(candts), k, replace=False)
    candts = np.array(candts)
    return candts[indxs]


def del_edges_de_nd(graph, node_labels, k=1):
    """ Disconnect externally strategy. Only nodes with different label are selected as candidates to remove. """
    # Set weight of 2 for edges with same label and 1 for the remaining ones
    nld = dict(node_labels)
    di = {(u, v): 1 if nld[u] == nld[v] else 2 for u, v in graph.edges}
    graph_ = graph.copy()
    nx.set_edge_attributes(graph_, values=di, name='weight')
    return _top_nonmst_filtr_edges(graph_, k, filter=1)


# ---------------------
# Edge addition attacks
# ---------------------

def add_edges_rand(graph, k=1):
    """ Adds at most k random edges to the graph. No overlap and self loop checks. """
    n = len(graph.nodes)
    edges = np.random.choice(n, size=2*k).reshape(k, 2)
    edges.sort()
    return edges


def add_edges_rand_no(graph, k=1):
    """ Add exactly k random edges to an undirected graph. Adds strictly new edges and no self loops. """
    n = len(graph.nodes())
    m = len(graph.edges())

    # Make sure we have enough non-edges
    if k > (n ** 2 - (m + n)):
        raise ValueError('Too many edges required!')

    e_lindx = set(np.ravel_multi_index(triu(nx.adj_matrix(graph), k=1).nonzero(), (n, n)))

    edges = []
    while len(edges) < k:
        edge = list(np.random.randint(0, n, size=2))
        edge.sort()
        if edge[0] == edge[1]:
            edge = [edge[0], edge[1]+1]
        li = np.ravel_multi_index(edge, (n, n))
        if li not in e_lindx:
            e_lindx.add(li)
            edges.append(edge)

    return np.array(edges)


def add_edges_deg(graph, k=1):
    """ Add edges using a degree attachment strategy. No overlap and self loop checks. """
    # Sample a set of edges uniformly and from each an end node with equal probability
    # The selected nodes will have been chosen proportional to their degrees
    # Select pairs of nodes and connect them to each other
    edges = np.array(list(graph.edges()))
    e_indx = np.random.randint(0, len(edges), k*2)
    v_indx = np.random.randint(0, 2, k*2)       # Uniform 0,1 sampling to select the end node
    verts = edges[e_indx, v_indx]
    return zip(verts[:k], verts[k:])


def add_edges_deg_no(graph, k=1):
    """ Add edges using a degree attachment strategy. Adds strictly new edges and no self loops. """
    # Sample edges uniformly and from each an end node with equal probability
    # The selected nodes will have been chosen proportional to their degrees
    # Select pairs of nodes and connect them to each other
    n = len(graph.nodes())
    m = len(graph.edges())

    # Make sure we have enough non-edges
    if k > (n ** 2 - (m + n)):
        raise ValueError('Too many edges required!')

    e_lindx = set(np.ravel_multi_index(triu(nx.adj_matrix(graph), k=1).nonzero(), (n, n)))
    edges = np.array(list(graph.edges()))

    new_edges = []
    while len(new_edges) < k:
        e_indx = np.random.randint(0, len(edges), 2)
        v_indx = np.random.randint(0, 2, 2)
        edge = edges[e_indx, v_indx]
        edge.sort()
        if edge[0] != edge[1]:
            li = np.ravel_multi_index(edge, (n, n))
            if li not in e_lindx:
                e_lindx.add(li)
                new_edges.append(edge)

    return np.array(new_edges)


def add_edges_pa(graph, k=1):
    """ Add edges using a preferential attachment strategy. No overlap and self loop checks.
    See: https://www.cs.purdue.edu/homes/neville/courses/NetworkSampling-KDD13-final.pdf """
    # Sample a set of nodes uniformly at random and connect them to other nodes sampled based on degree.
    # To get the degree based samples take a set of edges uniformly and from each an end node with equal probability
    # The selected nodes will have been chosen proportional to their degrees
    src = np.random.choice(graph.nodes(), k)
    edges = np.array(list(graph.edges()))
    e_indx = np.random.randint(0, len(edges), k)
    v_indx = np.random.randint(0, 2, k)
    dst = edges[e_indx, v_indx]
    return zip(src, dst)


def add_edges_pa_no(graph, k=1):
    """ Add edges using a preferential attachment strategy. Adds strictly new edges and no self loops. """
    # Sample a set of nodes uniformly at random and connect them to other nodes sampled based on degree.
    # To get the degree based samples take a set of edges uniformly and from each an end node with equal probability
    # The selected nodes will have been chosen proportional to their degrees
    n = len(graph.nodes())
    m = len(graph.edges())

    # Make sure we have enough non-edges
    if k > (n ** 2 - (m + n)):
        raise ValueError('Too many edges required!')

    e_lindx = set(np.ravel_multi_index(triu(nx.adj_matrix(graph), k=1).nonzero(), (n, n)))
    edges = np.array(list(graph.edges()))

    new_edges = []
    while len(new_edges) < k:
        src = np.random.choice(graph.nodes())
        e_indx = np.random.randint(0, len(edges))
        v_indx = np.random.randint(0, 2)
        dst = edges[e_indx, v_indx]
        edge = [src, dst]
        edge.sort()
        if edge[0] != edge[1]:
            li = np.ravel_multi_index(edge, (n, n))
            if li not in e_lindx:
                e_lindx.add(li)
                new_edges.append(edge)

    return np.array(new_edges)


def add_edges_da_no(graph, k=1):
    """ Add edges that increase the assortativity of the network. Adds strictly new edges and no self loops. """
    # Two options:
    # 1) greedy: add edges that will increase assortativity the most.
    # 2) random: add edges that will increase assortativity.
    # In 1) we select src nodes according to how far they are from the mean, in 2) we selected them uniformly at random
    n = len(graph.nodes())
    e_lindx = set(np.ravel_multi_index(triu(nx.adj_matrix(graph), k=1).nonzero(), (n, n)))

    deg = np.array(graph.degree())
    deg = deg[deg[:, 0].argsort()]
    mu = np.mean(deg[:, 1])
    dev = np.abs(deg[:, 1] - mu) + 1
    cs = np.cumsum(dev)

    edges = []
    while len(edges) < k:
        # Option 1) Sample a src node proportional to its distance from mu
        idx_src = np.searchsorted(cs, np.random.randint(0, cs[-1]))
        src = deg[idx_src, 0]
        # Option 2) Alternatively sample src uniformly at random
        # src = np.random.choice(graph.nodes())

        # Sample a dst with similar deg to src
        mu_src = deg[idx_src, 1]
        dev = np.abs(deg[:, 1] - mu_src) + 1
        dev = 1 / dev

        # Ensure we cannot sample the src again nor any node on the other side of mu
        dev[src] = 0
        if mu_src > mu:
            dev[deg[:, 1] < mu] = 0
        elif mu_src < mu:
            dev[deg[:, 1] > mu] = 0
        else:
            dev[deg[:, 1] != mu] = 0

        cs_src = np.cumsum(dev)
        if cs_src[-1] == 0:
            continue
        idx_dst = np.searchsorted(cs_src, np.random.randint(0, cs_src[-1]))
        dst = deg[idx_dst, 0]

        # Create edge and check if already exists
        edge = [src, dst]
        edge.sort()
        li = np.ravel_multi_index(edge, (n, n))
        if li not in e_lindx:
            e_lindx.add(li)
            edges.append(edge)

    return np.array(edges)


def add_edges_dd_no(graph, k=1):
    """ Add edges that increase the disassortativity of the network. Adds strictly new edges and no self loops. """
    n = len(graph.nodes())
    e_lindx = set(np.ravel_multi_index(triu(nx.adj_matrix(graph), k=1).nonzero(), (n, n)))

    deg = np.array(graph.degree())
    deg = deg[deg[:, 0].argsort()]
    mu = np.mean(deg[:, 1])
    dev = np.abs(deg[:, 1] - mu) + 1
    cs = np.cumsum(dev)

    edges = []
    while len(edges) < k:
        # Sample a src node proportional to its distance from mu
        idx_src = np.searchsorted(cs, np.random.randint(0, cs[-1]))
        src = deg[idx_src, 0]

        # Sample a dst with as dissimilar deg to src as possible
        mu_src = deg[idx_src, 1]
        dev = np.abs(deg[:, 1] - mu_src) + 1

        # Ensure we cannot sample the src again nor any node on the other side of mu
        dev[src] = 0
        if mu_src > mu:
            dev[deg[:, 1] > mu] = 0
        elif mu_src < mu:
            dev[deg[:, 1] < mu] = 0
        else:
            dev[deg[:, 1] != mu] = 0

        cs_src = np.cumsum(dev)
        if cs_src[-1] == 0:
            continue
        idx_dst = np.searchsorted(cs_src, np.random.randint(0, cs_src[-1]))
        dst = deg[idx_dst, 0]

        # Create edge and check if already exists
        edge = [src, dst]
        edge.sort()
        li = np.ravel_multi_index(edge, (n, n))
        if li not in e_lindx:
            e_lindx.add(li)
            edges.append(edge)

    return np.array(edges)


def add_edges_ci_no(graph, node_labels, k=1):
    """ Add edges using connect internally. Edges are added only between nodes that share the same label.
     Only adds edges which do not exist already. """
    n = len(graph.nodes())
    e_lindx = set(np.ravel_multi_index(triu(nx.adj_matrix(graph), k=1).nonzero(), (n, n)))

    nl = np.array(node_labels)
    nl = nl[nl[:, 1].argsort()]
    group_lbls = np.split(nl[:, 0], np.unique(nl[:, 1], return_index=True)[1][1:])

    edges = []
    while len(edges) < k:
        lbl_arr = np.random.choice(group_lbls)
        if len(lbl_arr) > 2:
            edge = list(np.random.choice(lbl_arr, 2, replace=False))
            edge.sort()
            # get linear index and check the edge does not exist
            li = np.ravel_multi_index(edge, (n, n))
            if li not in e_lindx:
                e_lindx.add(li)
                edges.append(edge)

    return np.array(edges)


def add_edges_ce_no(graph, node_labels, k=1):
    """ Add edges using connect externally. Edges are added only between nodes that do not share the same label.
    Only adds edges which do not exist already. """
    n = len(graph.nodes())
    e_lindx = set(np.ravel_multi_index(triu(nx.adj_matrix(graph), k=1).nonzero(), (n, n)))

    nl = np.array(node_labels)
    nl = nl[nl[:, 1].argsort()]
    group_lbls = np.split(nl[:, 0], np.unique(nl[:, 1], return_index=True)[1][1:])

    edges = []
    while len(edges) < k:
        lbl_arrs = np.random.choice(group_lbls, 2, replace=False)
        edge = [np.random.choice(lbl_arrs[0]), np.random.choice(lbl_arrs[1])]
        edge.sort()
        # get linear index and check the edge does not exist
        li = np.ravel_multi_index(edge, (n, n))
        if li not in e_lindx:
            e_lindx.add(li)
            edges.append(edge)

    return np.array(edges)


# -------------
# Mixed attacks
# -------------

def dice(graph, node_labels, k=1):
    """ DICE: Disconnect Internally Connect Externally. For each perturbation we randomly select if we want to add or
    remove an edge. Edges are only removed between nodes of the same class and additions are only done between edges of
    different classes.

    Notes
    -----
    For this method we use teh definition provided in [1].
    [1] Daniel Zügner and Stephan Günnemann. 2019. Adversarial attacks on graph neural networks via meta learning. In
    Proceedings of the International Conference on Learning Representations.
    """
    op_ind = np.random.randint(0, 2, size=k)
    num_add = np.sum(op_ind)
    num_del = k - num_add

    # Call func to add edges
    add_edges = [] if num_add == 0 else add_edges_ce_no(graph, node_labels, num_add)
    del_edges = [] if num_del == 0 else del_edges_di_nd(graph, node_labels, num_del)

    return add_edges, del_edges


def fasttack(graph, k=1):
    pass


def rewire_rand(graph, k=1):
    """ Select k edges and randomly rewire them. Deletion is done without disconnect and addition without overlap. """
    ns = set(graph.nodes)

    # Delete k random edges
    del_edges = del_edges_rand_nd(graph, k=k)

    # Reconnect the source nodes to other nodes not in their neighbourhood nor themselves
    add_edges = map(lambda x: [x[0], np.random.choice(list(ns - set(list(graph.neighbors(x[0])) + [x[0]])))], del_edges)

    return np.array(add_edges), np.array(del_edges)


def rewire_da(graph, k=1):
    """ Rewire the network in order to increase degree assortativity. """
    pass


def rewire_dd(graph, k=1):
    """ Rewire the network in order to increase degree disassortativity. """
    pass
