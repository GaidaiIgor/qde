# QDE
Solves systems of ordinary first-order nonlinear differential equations by reformulating them in terms of Quadratic Unconstrained Binary Optimization (QUBO) problems, which allows to obtain solution on quantum annealers. Contains example application to molecular dynamics.

# Running
No installation is required. To run the code, simply run `test_core.py` with a Python 3 interpreter, which will propagate a simple example of a Hydrogen vibrational trajectory with the default options. Alternative options can be provided as arguments to `get_solution` in `main` function of `test_core.py`. 

A new problem can be defined by modifying `get_problem` function in `test_core.py`. See the existing definitions and the docstring comments for more details on this.
