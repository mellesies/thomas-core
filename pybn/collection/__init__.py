# -*- coding: utf-8 -*-

def parse_query_string(query_string):
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

        return dist, values

    query_str, given_str = query_string, ''

    if '|' in query_str:
        query_str, given_str = query_string.split('|')

    return split(query_str) + split(given_str)


class ProbabilisticModel(object):

    def compute_posterior(self, query_dist, query_values, evidence_dist, 
        evidence_values, **kwargs):
        """Compute the probability of the query variables given the evidence.

        The query P(I,G=g1|D,L=l0) would imply:
            query_dist = ['I']
            query_values = {'G': 'g1'}
            evidence_dist = ('D',)
            evidence_values = {'L': 'l0'}

        :param tuple query_dist: Random variable to query
        :param dict query_values: Random variable values to query
        :param tuple evidence_dist: Conditioned on evidence
        :param dict evidence_values: Conditioned on values
        :return: pandas.Series (possibly with MultiIndex)
        """
        raise NotImplementedError

    def P(self, query_string):
        """Return the probability as queried by query_string.

        P('I,G=g1|D,L=l0') is equivalent to calling compute_posterior with:
            query_dist = ('I',)
            query_values = {'G': 'g1'}
            evidence_dist = ('D',)
            evidence_values = {'L': 'l0'}
        """
        qd, qv, gd, gv = parse_query_string(query_string)
        return self.compute_posterior(qd, qv, gd, gv)

    def MAP(self, query_dist, evidence_values, include_probability=True):
        """Perform a Maximum a Posteriori query."""
        d = self.compute_posterior(query_dist, {}, [], evidence_values)
        evidence_vars = [e for  e in evidence_values.keys() if e in d.scope]

        d = d.droplevel(evidence_vars)

        if include_probability:
            return d.idxmax(), d.max()

        return d.idxmax()

