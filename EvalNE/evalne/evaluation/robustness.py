#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:
# Contact:
# Date: 25/04/2022

# This file contains methods and classes for simulating attacks on networks and evaluating robustness.

import numpy as np
from evalne.methods import edge_attacks as atks


class Attack(object):

    def __init__(self, graph, strategy, max_budget=0.01, approx=np.inf, node_labels=None):
        self._graph = graph
        self._n = len(self._graph.node)
        self._m = len(self._graph.edges)
        self._strategy = strategy
        self._max_budget = max_budget  # can be an int or a float
        self._approx = approx
        self._node_labels = node_labels
        self._edges = self._compute_edges()

    def _budget_to_k(self, budget):
        if budget < 1:
            return int(budget * self._m)
        else:
            return budget

    def _requires_labels(self):
        if self._strategy in ['del_edges_di', 'del_edges_di_nd', 'del_edges_de', 'del_edges_de_nd',
                              'add_edges_ci_no', 'add_edges_ce_no', 'dice']:
            return True
        else:
            return False

    def _compute_edges(self):
        func = getattr(atks, str(self._strategy))
        max_k = self._budget_to_k(self._max_budget)
        kwargs = {}
        if self._strategy in ['del_edges_ib', 'del_edges_ib_nd'] and self._approx is not None:
            kwargs.update({'approx': self._approx})

        if self._node_labels is not None and self._requires_labels():
            kwargs.update({'node_labels': self._node_labels})

        edges = func(graph=self._graph, k=max_k, **kwargs)
        return edges

    def attack(self, budget):
        if budget > self._max_budget:
            raise ValueError("The budget must be smaller or equal than max_budget!")
        else:
            k = self._budget_to_k(budget)
            graph_ = self._graph.copy()
            if 'del' in self._strategy:
                edges = self._edges[:k]
                graph_.remove_edges_from(edges)
            elif 'add' in self._strategy:
                edges = self._edges[:k]
                graph_.add_edges_from(edges)
            elif 'rewire' in self._strategy or self._strategy == 'dice':
                add_edges = self._edges[0][:k]
                del_edges = self._edges[1][:k]
                graph_.remove_edges_from(del_edges)
                graph_.add_edges_from(add_edges)
            else:
                raise ValueError('Unknown strategy {}'.format(self._strategy))
            return graph_

    def update_max_budget(self, max_budget):
        self._max_budget = max_budget
        self._edges = self._compute_edges()
