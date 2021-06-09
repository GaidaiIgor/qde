import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import qde
from qpsolvers import solve_qp

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


def test_qbsolv():
    """Dummy test to make sure QBSolv works."""
    Q = {('q1', 'q1'): 0.1, ('q2', 'q2'): 0.1, ('q1', 'q2'): -0.2}
    res = QBSolv().sample_qubo(Q)
    print(res)


def test_qubo(grid_from=0, grid_to=1, N=11, y1=1, qbits_integer=3, qbits_decimal=30, num_repeats=200, **kwargs):
    """Solves dy/dx = exp(x) by formulating it as a QUBO problem (in binary coefficients)."""
    grid = np.linspace(grid_from, grid_to, N)
    f = np.exp(grid)
    dx = grid[1] - grid[0]
    solution, error = qde.solve(f, dx, y1, qbits_integer, qbits_decimal, num_repeats=num_repeats, **kwargs)
    print('Solution:')
    print(solution)
    print(f'Error: {error}')


def test_matplotlib():
    """Dummy test to make sure matplotlib works."""
    plt.plot([1, 2, 3])
    plt.show()


def apply_plot_settings(axes):
    """Applies common settings."""
    axes.autoscale()
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlabel('N')
    axes.set_ylabel('Error')


def get_axes(axes):
    """If None, creates a new plot, otherwise returns its argument."""
    if axes is None:
        _, axes = plt.subplots()
        apply_plot_settings(axes)
    return axes


def filter_kwargs(func, kwargs):
    """Returns kwargs subset where only keys recognized by func are left."""
    return {key: value for key, value in kwargs.items() if key in func.__code__.co_varnames}


def get_plot_keys():
    """Returns known plot keys."""
    return {'color', 'linestyle', 'linewidth',  'marker', 'markersize', 'label'}


def filter_kwargs_plot(kwargs):
    """Filters out keys unknown to plot function."""
    known_keys = get_plot_keys()
    return {key: value for key, value in kwargs.items() if key in known_keys}


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
