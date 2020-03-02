[![Coverage Status](https://coveralls.io/repos/github/mellesies/py-bn/badge.svg?branch=master)](https://coveralls.io/github/mellesies/thomas-core?branch=thomas)
[![Build Status](https://travis-ci.org/mellesies/thomas-core.svg?branch=thomas)](https://travis-ci.org/mellesies/thomas-core)

# Thomas
Very simple (almost naive ;-) bayesian network implementation.

Example (module `thomas.core.examples`) contains examples from the book "Probabilistic Graphical Models: Principles and Techniques" from Koller and Friedman ([PGM Stanford](http://pgm.stanford.edu)) and from the lecture by [Adnan Darwiche](http://web.cs.ucla.edu/~darwiche/) on YouTube:
* [6a. Inference by Variable Elimination I (Chapter 6)](https://www.youtube.com/watch?v=7oRReD_ayWo).
* [6b. Inference by Variable Elimination II (Chapter 6)](https://www.youtube.com/watch?v=QSSmx1ndUvg).

For information on how to setup a network based on (known) conditional probabilities see `pybn.examples.get_student_network()` or have a look at the [examples.ipynb](examples.ipynb) notebook.

To get started with querying a network, try the following:
```python
from thomas.core import examples

# Load an example network
Gs = examples.get_student_network()

# This should output the prior probability of random variable 'S' (SAT score).
print(Gs.P('S'))
print()

# Expected output:
# P(S)
# S
# s0    0.725
# s1    0.275
# dtype: float64

# Query for the conditional probability of S given the student is intelligent.
print(Gs.P('S|I=i1'))

# Expected output:
# P(S)
# S
# s0    0.2
# s1    0.8
# dtype: float64
```

For details have a look at the notebooks, code or use pydoc.
