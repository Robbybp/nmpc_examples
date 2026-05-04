# NMPC Examples

This repository was intended to provide examples of using specialized data
structures for nonlinear model predictive control simulations with Pyomo
models. While the examples here may still be useful, **the functionality
contained in the `nmpc_examples/nmpc` subdirectory has been superceded** by the
MPC Pyomo extension, with code
[here](https://github.com/Pyomo/pyomo/tree/main/pyomo/contrib/mpc)
and documentation
[here](https://pyomo.readthedocs.io/en/stable/contributed_packages/mpc/index.html).
For a more complicated example using the `pyomo.contrib.mpc` software, please
see [this code](https://github.com/IDAES/publications/tree/main/parker_jpc2023),
which produces the chemical looping case study results for the paper referenced
below.

## Citation

If you find these examples or data structures useful for research, please cite
the following paper, which describes the motivation for these data structures
and some of the underlying Pyomo features that make this possible.
```bibtex
@article{parker2023mpc,
title = {Model predictive control simulations with block-hierarchical differential-algebraic process models},
journal = {Journal of Process Control},
volume = {132},
pages = {103113},
year = {2023},
issn = {0959-1524},
doi = {https://doi.org/10.1016/j.jprocont.2023.103113},
url = {https://www.sciencedirect.com/science/article/pii/S0959152423002007},
author = {Robert B. Parker and Bethany L. Nicholson and John D. Siirola and Lorenz T. Biegler},
}
```
