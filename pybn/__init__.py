# -*- coding: utf-8 -*-
"""Module with classes that facilitate building a Bayesian network.

Examples are borrowed from Koller and Friedmand's "Probabilistic 
Graphical Models: Principles and Techniques"
"""

from collections import OrderedDict

import numpy as np
import pandas as pd


class BayesianNetwork(object):
    """Bayesian Network."""

    def __init__(self, name, nodes):
        """Initialize a new Bayesian Network.

        :param str name: Name of the Network
        :param dict nodes: Dictionary of Node instances, indexed by 
                           name of the random variable.
        """
        if not isinstance(nodes, (dict, OrderedDict)):
            raise Exception('nodes should be a dict!')
        
        self.name = name
        self.nodes = OrderedDict(nodes.items())
        self.JPD = self.compute_joint_distribution()
        # self.JPD = self.JPD.sort_index()
    
    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        s = '<BayesianNetwork: {}>'.format(self.name)
        for n in self.nodes.values():
            s += '\n' + repr(n)
        return s
    
    def get_parameter_space(self, rv=None, full=True):
        """Return the parameter space for a random variable (RV) or the entire
        network. The parameter space for a single RV is the 
        list of possible outcomes for that variable.

        :param str rv: Name of the RV to return the parameter space for.
                       If None, the network's entire space is returned.
        :param bool full: Indicates whether to include all parents (full=True) 
                          or only the direct parents (full=False) of the RV.
        :return: tuple where t[0] yields the parameter space for the 
        requested RV and t[1] is a dict of spaces, indexed by RV name.
        """
        if rv is None:
            nodes = list(self.nodes.values())
            rv_space = []
        else:
            rv_space = list(self.nodes[rv].CPD.columns)      
            if full:
                nodes = self.nodes[rv].parents
            else:
                nodes = self.nodes[rv].direct_parents
        
        parent_space = dict()
                
        for n in nodes:
            parent_space[n.RV] = list(n.CPD.columns)
        
        return rv_space, parent_space
    
    def compute_parameter_combinations(self, rv=None):
        """Return a list of parameter combinations for a random variable (RV).

        :param str rv: Name of the random variable to return the parameter
                       combinations for or None to return the combinations
                       for the entire network.
        :return: list of parameter combinations.
        """
        rv_space, parent_space = self.get_parameter_space(rv)

        if rv_space:
            space = list(parent_space.values()) + [rv_space, ]
        else:
            space = list(parent_space.values())
                
        return list(pd.MultiIndex.from_product(space.values()))

    def compute_joint_distribution(self):
        """Compute the Joint Distribution for all random variables
        in the network.

        :return: MultiIndexed pandas.Series()
        """
        _, space = self.get_parameter_space()
        idx = pd.MultiIndex.from_product(space.values(), names=space.keys())
        
        # Compute the probability of each of the combinations.
        # Probability equals node * node * node * ... 
        probabilities = []
        
        for combination in idx:
            zipped_as_dict = dict(zip(idx.names, combination))
            
            ps = []
            for node in self.nodes.values():
                p = self.lookup_conditional_probability(node.RV, zipped_as_dict[node.RV], **zipped_as_dict)
                ps.append(float(p))
            
            probability = np.prod(ps)
            probabilities.append(probability)
        
        return pd.Series(probabilities, index=idx)
        
    def compute_prior(self, query_dist, query_values=None):
        """Compute the prior probability.

        :param tuple query_dist: Names of the random variables to compute the prior for.
        :param dict query_values: Values to compute the prior for.
        :return: pandas.Series
        """
        if not isinstance(query_dist, (tuple, list)):
            query_dist = (query_dist, )

        if query_values is None:
            query_values = dict()

        # query = tuple('{}={}'.format(key,val) for key,val in query_values.items())
        # query = query + query_dist
        # print("P({})".format(','.join(query)))

        sum_level = tuple(query_values.keys()) + query_dist
        JPD = self.JPD.sum(level=sum_level)

        if query_values:
            idx = tuple(query_values.values())
            idx_level = tuple(query_values.keys())
            
            if isinstance(JPD.index, pd.MultiIndex):
                JPD = JPD.xs(idx, level=idx_level)
                JPD = JPD.sort_index()
            else:
                JPD = JPD[idx[0]]

        return JPD

    def compute_posterior(self, query_dist, query_values, evidence_dist, evidence_values):
        """Compute the probability of the query variables given the evidence.
        
        The query P(I,G=g1|D,L=l0) would imply:
            query_dist = ('I',)
            query_values = {'G': 'g1'}
            evidence_dist = ('D',)
            evidence_values = {'L': 'l0'}

        :param tuple query_dist: Random variable to query
        :param dict query_values: Random variable values to query
        :param tuple evidence_dist: Conditioned on evidence
        :param dict evidence_values: Conditioned on values
        :return: pandas.Series (possibly with MultiIndex)
        """
        if not isinstance(query_dist, (tuple, list)):
            query_dist = (query_dist, )

        if not (evidence_dist or evidence_values):
            return self.compute_prior(query_dist, query_values)

        query = tuple('{}={}'.format(key,val) for key,val in query_values.items()) 
        query = query + query_dist

        given = evidence_dist + tuple('{}={}'.format(query_dist,val) for query_dist,val in evidence_values.items())
        print("P({}|{})".format(','.join(query), ','.join(given)))

        for rv in query_dist:
            if rv in evidence_dist + tuple(evidence_values.keys()):
                raise Exception('overlap!') 

        # Determine which random variables occur in the denominator
        #  --> the variables we're conditioning upon
        levels_denominator = evidence_dist + tuple(evidence_values.keys())

        # The numerator additionally contains the random variables we're querying
        levels_numerator = levels_denominator + tuple(query_values.keys()) + query_dist

        # Note that the order of the variables needs to match!
        numerator = self.JPD.sum(level=levels_numerator).sort_index()
        denominator = self.JPD.sum(level=levels_denominator).sort_index()

        # Iterate over the index of the denominator: 
        #  - nominator[idx] yields a list/array/series with a single level
        #  - denominator[idx] yields a scalar.
        #  - division array/scalar --> array
        # All results are appended to a single list
        s = list()

        for idx in denominator.index:
            s += list(numerator[idx] / denominator[idx])

        # Cast the result to a series and set the index to the (multilevel) 
        # index of the numerator.
        CPD = pd.Series(s, index=numerator.index)

        # Filter the CPD based on the parameters provided
        if query_values or evidence_values:
            levels = tuple(query_values.keys()) + tuple(evidence_values.keys())
            values = tuple(query_values.values()) + tuple(evidence_values.values())
            CPD = CPD.xs(values, level=levels)

        return CPD

    def parse_query_string(self, query_string):
        """Parse a query string into a tuple of query_dist, query_values, 
        evidence_dist, evidence_values.

        The query P(I,G=g1|D,L=l0) would imply:
            query_dist = ('I',)
            query_values = {'G': 'g1'}
            evidence_dist = ('D',)
            evidence_values = {'L': 'l0'}        
        """
        def split(s):
            dist, values = [], {}
            params = []

            if s:
                params = s.split(',')

            for p in params:
                if '=' in p:
                    key, value = p.split('=')
                    values[key] = value
                else:
                    dist.append(p)

            return tuple(dist), values

        query_str, given_str = query_string, ''

        if '|' in query_str:
            query_str, given_str = query_string.split('|')

        return split(query_str) + split(given_str)

    def P(self, query_string):
        """Return the probability as queried by query_string.

        P('I,G=g1|D,L=l0') is equivalent to calling compute_posterior with:
            query_dist = ('I',)
            query_values = {'G': 'g1'}
            evidence_dist = ('D',)
            evidence_values = {'L': 'l0'}                    
        """
        qd, qv, gd, gv = self.parse_query_string(query_string)
        return self.compute_posterior(qd, qv, gd, gv)

    def propagate(self, **kwargs):
        """Compute/print all posterior probabilities given evidence provided
        by **kwargs.

        e.g. propagate(I='i0') will print tables for all random variables in
        the network.
        """
        for RV, node in self.nodes.items():
            print('-' * 80)
            print('{}: {}'.format(node.name, node.query_string))


            print('prior:')
            print(self.compute_prior(node.RV, {}))
            print()

            try:
                print('posterior:')
                print(self.compute_posterior(node.RV, {}, (), kwargs))
            except Exception as e:
                print('Cannot compute :-(')
                print('Error:', e)

            print()

    def lookup_conditional_probability(self, rv, value, **kwargs):
        """Return the value in the CPD for a specific random variable.

        Convenience method.
        """
        if rv in kwargs:
            del kwargs[rv]
            
        rv_space, parent_space = self.get_parameter_space(rv, full=False)

        if value not in rv_space:
            msg = "value '{}' is not one of {}".format(value, rv_space)
            raise Exception(msg) 
        
        CPD = self.nodes[rv].CPD
        given = tuple(kwargs.get(x) for x in CPD.index.names if x is not None)
        
        if given:
            return CPD.loc[given, value]
            
        return CPD.loc[:, value]
        

class Node(object):
    def __init__(self, RV, name, CPD=None):
        """
        :param str RV: Name of the random variable
        :param str name: Name of the Node 
        :param pandas.DataFrame CPD: Conditional Probability Distribution

        if CPD is provided rows should be indexed by potential values of the
        parent's distribution and columns should be index by the RV's space.

        E.g. P(S|I):
        S     s0    s1
        I             
        i0  0.95  0.05
        i1  0.20  0.80
        """
        self.RV = RV
        self.name = name
        self.CPD = CPD
        
        self.direct_parents = []
        self.direct_children = []
    
    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        values = list(self.CPD.columns)
        return "Node '{}' ({}): {}".format(self.RV, self.name, values)
    
    @property
    def query_string(self):
        """Return the query string associated with this Node's CPD."""
        names = [n for n in self.CPD.index.names if n is not None]
        if names:
            return 'P({}|{})'.format(self.RV, ','.join(names))
        return 'P({})'.format(self.CPD.columns.names[0])

    @property
    def conditioning(self):
        """Return the list of values that condition this Node's CPD."""
        return tuple([n for n in self.CPD.index.names if n is not None])

    @property
    def conditioned(self):
        """Return the list of values that are conditioned this Node's CPD."""
        return self.CPD.columns.names[0]

    @property
    def parents(self):
        """Return *all* parents (recursively)."""
        dpp = []
        for p in self.direct_parents:
            dpp += p.parents
        
        return self.direct_parents + dpp
    
    @property
    def children(self):
        """Return *all* children (recursively)."""
        dcc = []
        for c in self.direct_children:
            dcc += c.children
        
        return self.direct_children + dcc
        
    def add_child(self, c):
        """Add a child to this Node.
        :param Node c: child to add.
        """
        if c in self.parents:
            raise Exception('provided child is already a (direct) parent!?')
        
        c.direct_parents.append(self)
        self.direct_children.append(c)

    

from . import examples
