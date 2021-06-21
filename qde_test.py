import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import qde
from qpsolvers import solve_qp

from plots_general import myplot

import addcopyfighandler


def get_problem(problem, grid):
    """Returns problem-specific values: funcs, boundary condition and answer for a given problem and grid."""
    if problem == 0:
        # Problem: dy/dx = exp(x)
        # Solution: y(x) = exp(x)
        funcs = np.zeros((3, grid.shape[0]))
        funcs[0, :] = -np.exp(grid)
        funcs[2, :] = 1
        known_points = np.exp(grid[0:1])
        ans = np.exp(grid[-1])
    else:
        raise Exception('Unknown problem')

    return funcs, known_points, ans


def test_qp(grid_from=0, grid_to=1, N=11, deriv=lambda xs: np.exp(xs), y1=1, ans=np.exp(1), **kwargs):
    """Solves given differential equation by formulating it as a Quadratic Programming (QP) problem (real coefficients)."""
    grid = np.linspace(grid_from, grid_to, N)
    f = deriv(grid)
    dx = grid[1] - grid[0]
    H = qde.build_quadratic_minimization_matrix(len(f), dx)
    d = qde.build_quadratic_minimization_vector(f, dx, y1)

    solution = np.concatenate(([y1], solve_qp(2 * H, d)))
    print('Solution:')
    print(solution)
    error = abs(solution[-1] - ans)
    print(f'Error: {error}')
    return solution


def test_qp_general(grid_from=0, grid_to=1, N=11, problem=0, **kwargs):
    """Solves given differential equation by formulating it as a Quadratic Programming (QP) problem (real coefficients). Builds matrices with general algorithm."""
    grid = np.linspace(grid_from, grid_to, N)
    funcs, known_points, ans = get_problem(problem, grid)
    dx = grid[1] - grid[0]
    H, d = qde.build_quadratic_minimization_matrices_general(funcs, dx, known_points, max_considered_accuracy=2, points_per_step=N)

    solution = np.concatenate((known_points, solve_qp(2 * H, d)))
    print('Solution:')
    print(solution)
    error = abs(solution[-1] - ans)
    print(f'Error: {error}')
    return solution


def test_qubo(grid_from=0, grid_to=1, N=11, deriv=lambda xs: np.exp(xs), y1=1, ans=np.exp(1), qbits_integer=3, qbits_decimal=30, num_repeats=200, **kwargs):
    """Solves given differential equation by formulating it as a Quadratic Unconstrained Binary Optimization (QUBO) problem (binary coefficients)."""
    grid = np.linspace(grid_from, grid_to, N)
    f = deriv(grid)
    dx = grid[1] - grid[0]
    solution, error_sln = qde.solve(f, dx, y1, qbits_integer, qbits_decimal, num_repeats=num_repeats, **kwargs)
    print('Solution:')
    print(solution)
    print(f'Solution error: {error_sln}')
    error = abs(solution[-1] - ans)
    print(f'Error: {error}')
    return solution


def plot_h2_potential():
    kg_per_amu = 1.660538921e-27
    kg_per_aum = 9.10938291e-31
    aum_per_amu = kg_per_amu / kg_per_aum
    mass = 1.00782503207 * aum_per_amu


def plot_qp_error(grid_from=0, grid_to=1, problem=0, **kwargs):
    """Plots QP solution error as a function of number of grid points."""
    Ns = np.geomspace(10, 100, 5, dtype=int)
    plot_data = np.empty((2, len(Ns)))
    for i in range(len(Ns)):
        N = Ns[i]
        grid = np.linspace(grid_from, grid_to, N)
        funcs, known_points, ans = get_problem(problem, grid)
        dx = grid[1] - grid[0]
        H, d = qde.build_quadratic_minimization_matrices_general(funcs, dx, known_points, points_per_step=N, **kwargs)
        solution = np.concatenate((known_points, solve_qp(2 * H, d)))
        error = abs(solution[-1] - ans)
        plot_data[:, i] = (N, error)

    return myplot(plot_data[0, :], plot_data[1, :], **kwargs)


def plot_qubo_error(grid_from=0, grid_to=1, Ns=np.geomspace(10, 100, 5, dtype=int), deriv=lambda xs: np.exp(xs), y1=1, ans=np.exp(1), qbits_integer=3, qbits_decimal=30, num_repeats=200,
                    label='QUBO', **kwargs):
    """Plots QUBO error as a function of number of grid points."""
    plot_data = np.empty((2, len(Ns)))
    for i in range(len(Ns)):
        N = Ns[i]
        grid = np.linspace(grid_from, grid_to, N)
        f = deriv(grid)
        dx = grid[1] - grid[0]

        solution_bin, error_sln = qde.solve(f, dx, y1, qbits_integer, qbits_decimal, num_repeats=num_repeats, **kwargs)
        print(solution_bin[-1])
        print(error_sln)
        error = abs(solution_bin[-1] - ans)
        plot_data[:, i] = (N, error)

    return myplot(plot_data[0, :], plot_data[1, :], label=label, **kwargs)


if __name__ == '__main__':
    np.set_printoptions(precision=15, linewidth=200)
    test_qp_general(N=4)
    if not mpl.is_interactive():
        plt.show()
