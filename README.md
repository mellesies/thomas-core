[![Coverage Status](https://coveralls.io/repos/github/mellesies/py-bn/badge.svg?branch=master)](https://coveralls.io/github/mellesies/thomas-core?branch=thomas)
[![Build Status](https://travis-ci.org/mellesies/thomas-core.svg?branch=thomas)](https://travis-ci.org/mellesies/thomas-core)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mellesies/thomas-core/thomas?urlpath=/lab/tree/notebooks)

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

Alternatively, you can have a go at the example notebooks through [Binder](https://mybinder.org):
* [notebooks/1. Factors.ipynb](https://mybinder.org/v2/gh/mellesies/thomas-core/thomas?filepath=notebooks%2F1.%20Factors.ipynb)
* [notebooks/2. Bags of factors.ipynb](https://mybinder.org/v2/gh/mellesies/thomas-core/thomas?filepath=notebooks%2F2.%20Bags%20of%20factors.ipynb)
* [notebooks/3. Conditional probability tables.ipynb](https://mybinder.org/v2/gh/mellesies/thomas-core/thomas?filepath=notebooks%2F3.%20Conditional%20probability%20tables.ipynb)
* [notebooks/4. Bayesian Networks.ipynb](https://mybinder.org/v2/gh/mellesies/thomas-core/thomas?filepath=notebooks%2F4.%20Bayesian%20Networks.ipynb)
