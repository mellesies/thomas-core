# py-bn
Very simple (almost naive ;-) bayesian network implementation.

Examples (`pybn.setup_conditional_distributions()` and `pybn.setup_student_network()`) are based on the book "Probabilistic Graphical Models: Principles and Techniques" from Koller and Friedman ([PGM Stanford](http://pgm.stanford.edu)).

For information on how to setup a network based on (known) conditional probabilities see `pybn.setup_student_network()`.

To get started with querying a network, try the following:
```python
import pybn as bn

# Load an example network
Gs = bn.setup_student_network()

# This should output the prior probability of random variable 'S' (SAT score).
Gs.P('S')
# S
# s0    0.725
# s1    0.275
# dtype: float64

# Query for the conditional probability of S given the student is intelligent.
Gs.P('S|I=i1')
# S
# s0    0.2
# s1    0.8
# dtype: float64
```

For details have a look at the code or use pydoc.
