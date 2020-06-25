# -*- coding: utf-8 -*-
"""JPT: the Joint Probability Table"""

import numpy as np

from .base import ProbabilisticModel
from .factor import Factor
from .cpt import CPT

# ------------------------------------------------------------------------------
# JPT
# ------------------------------------------------------------------------------
class JPT(Factor, ProbabilisticModel):
    """Joint Probability Table."""

    @property
    def display_name(self):
        names = ','.join(self.scope)
        return f'JPT({names})'

    @classmethod
    def from_data(cls, df, cols=None, states=None, complete_value=0):
        """Create a full JPT from data (using Maximum Likelihood Estimation).

        Determine the empirical distribution by ..
          1. counting the occurrences of combinations of variable states; the
             heavy lifting is done by Factor.from_data().
          2. normalizing the result

        Note that the this will *drop* any NAs in the data.

        Args:
            df (pandas.DataFrame): data
            cols (list): columns in the data frame to use. If `None`, all
                columns are used.
            states (dict): list of allowed states for each random
                variable, indexed by name. If states is None, `jpt`
                should be a pandas.Series with a proper Index/MultiIndex.
            complete_value (int): Base (count) value to use for combinations of
                variable states in the dataset.

        Return:
            JPT (normalized)
        """
        factor = Factor.from_data(df, cols, states, complete_value)
        return JPT(factor.normalize().values, states=factor.states)

    def compute_dist(self, qd, ed=None):
        """Compute a (conditional) distribution.

        This is short for self.compute_posterior(qd, {}, ed, {})
        """
        if ed is None:
            ed = []

        return self.compute_posterior(qd, {}, ed, {})

    def compute_posterior(self, qd, qv, ed, ev):
        """Compute the (posterior) probability of query given evidence.

        The query P(I,G=g1|D,L=l0) would imply:
            qd = ['I']
            qv = {'G': 'g1'}
            ed = ['D']
            ev = {'L': 'l0'}

        Args:
            qd (list): query distributions: RVs to query
            qv (dict): query values: RV-values to extract
            ed (list): evidence distributions: coniditioning RVs to include
            ev (dict): evidence values: values to set as evidence.

        Returns:
            CPT

        """
        # Get a list of *all* variables to query
        query_vars = list(qv.keys()) + qd
        evidence_vars = list(ev.keys()) + ed

        result = self.project(query_vars + evidence_vars)

        if evidence_vars:
            result = result / result.sum_out(query_vars)

        # If query values were specified we can extract them from the factor.
        if qv:
            result = result.get(**qv)

        if isinstance(result, Factor):
            return CPT(result.values, states=result.states, conditioned=query_vars)

        elif isinstance(result, np.ndarray) and len(result) == 1:
            result = result[0]

        return result
