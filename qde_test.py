import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import qde
from qpsolvers import solve_qp

from plots_general import get_axes

import addcopyfighandler

def test_qp():
    """Solves dy/dx = exp(x) by formulating it as a Quadratic Programming problem (in real coefficients)."""
    grid_from = 0
    grid_to = 0.1
    N = 2
    y1 = 1

    grid = np.linspace(grid_from, grid_to, N)
    f = np.exp(grid)
    dx = grid[1] - grid[0]
    H = 2 * qde.build_quadratic_minimization_matrix(len(f), dx)
    d = qde.build_quadratic_minimization_vector(f, dx, y1)
    solution = solve_qp(H, d)
    print('Solution:')
    print(solution)


def test_qubo(grid_from=0, grid_to=1, N=11, y1=1, qbits_integer=3, qbits_decimal=30, num_repeats=200, **kwargs):
    """Solves dy/dx = exp(x) by formulating it as a QUBO problem (in binary coefficients)."""
    grid = np.linspace(grid_from, grid_to, N)
    f = np.exp(grid)
    dx = grid[1] - grid[0]
    solution, error = qde.solve(f, dx, y1, qbits_integer, qbits_decimal, num_repeats=num_repeats, **kwargs)
    print('Solution:')
    print(solution)
    print(f'Error: {error}')


def plot_qp_error(grid_from=0, grid_to=1, Ns=np.geomspace(10, 100, 5, dtype=int), y1=1, axes=None):
    """Plots QP solution error as a function of number of grid points."""
    axes = get_axes(axes)
    plot_data = np.empty((2, len(Ns)))
    for i in range(len(Ns)):
        N = Ns[i]
        grid = np.linspace(grid_from, grid_to, N)
        f = np.exp(grid)
        dx = grid[1] - grid[0]

        H = 2 * qde.build_quadratic_minimization_matrix(len(f), dx)
        d = qde.build_quadratic_minimization_vector(f, dx, y1)
        solution_real = solve_qp(H, d)
        error = abs(solution_real[-1] - np.exp(grid_to))
        plot_data[:, i] = (N, error)

    axes.plot(plot_data[0, :], plot_data[1, :], 'b.-', markersize=10, label='QP')
    return axes


def plot_qubo_error(grid_from=0, grid_to=1, Ns=np.geomspace(10, 100, 5, dtype=int), y1=1, qbits_integer=3, qbits_decimal=30, num_repeats=200, axes=None, **kwargs):
    """Plots QUBO error as a function of number of grid points."""
    axes = get_axes(axes)
    plot_data = np.empty((2, len(Ns)))
    for i in range(len(Ns)):
        N = Ns[i]
        grid = np.linspace(grid_from, grid_to, N)
        f = np.exp(grid)
        dx = grid[1] - grid[0]

        solution_bin, error_sln = qde.solve(f, dx, y1, qbits_integer, qbits_decimal, num_repeats=num_repeats, **kwargs)
        print(error_sln)
        error = abs(solution_bin[-1] - np.exp(grid_to))
        plot_data[:, i] = (N, error)

    axes.plot(plot_data[0, :], plot_data[1, :], markersize=10, **filter_kwargs_plot(kwargs))
    return axes


if __name__ == '__main__':
    np.set_printoptions(precision=15, linewidth=200)
    plot_solution_error()
    if not mpl.is_interactive():
        plt.show()
