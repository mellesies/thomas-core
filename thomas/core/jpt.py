# -*- coding: utf-8 -*-
"""JPT: the Joint Probability Table"""

from .base import ProbabilisticModel
from .factor import Factor
from .cpt import CPT

# ------------------------------------------------------------------------------
# JPT
# ------------------------------------------------------------------------------
class JPT(Factor, ProbabilisticModel):
    """JPT."""

    @property
    def display_name(self):
        names = [n for n in self._data.index.names if n is not None]
        names = ','.join(names)
        return f'JPT({names})'

    @classmethod
    def from_incomplete_jpt(cls, jpt, variable_states=None):
        """Create a full JPT from jpt that may not contain all combinations
        of variable states.

        Args:
            jpt (pandas.Series): jpt ...
            variable_states (dict): list of allowed states for each random
                variable, indexed by name. If variable_states is None, `jpt`
                should be a pandas.Series with a proper Index/MultiIndex.
        """
        if variable_states is None:
            # We'll need to try to determine variable_states from the jpt
            variable_states = dict(
                zip(jpt.index.names, jpt.index.levels)
            )

        # Create a factor containing *all* combinations set to 0
        f2 = Factor(0, variable_states)

        # By summing the Factor with the Series all combinations not in the
        # data are set to 0.
        return JPT(f2 + jpt)

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
            values = list(qv.values())

            if result.width == 1:
                result = result[values[0]]

            elif result.width > 1:
                indices = []

                for level, value in qv.items():
                    idx = result._data.index.get_level_values(level) == value
                    indices.append(list(idx))

                zipped = list(zip(*indices))
                idx = [all(x) for x in zipped]
                result = Factor(result._data[idx])


        if isinstance(result, Factor):
            result.sort_index()
            return CPT(result, conditioned_variables=query_vars)

        return result