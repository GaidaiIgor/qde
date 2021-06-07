import matplotlib.pyplot as plt
import numpy as np
import qde
from qpsolvers import solve_qp

import addcopyfighandler

def test_exp_qp():
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


def test_exp_qubo(grid_from = 0, grid_to = 1, N = 11, y1 = 1, qbits_integer = 2, qbits_decimal = 30, points_per_qubo = 1, num_repeats = 200):
    """Solves dy/dx = exp(x) by formulating it as a QUBO problem (in binary coefficients)."""
    grid = np.linspace(grid_from, grid_to, N)
    f = np.exp(grid)
    dx = grid[1] - grid[0]
    solution = qde.solve(f, dx, y1, qbits_integer, qbits_decimal, points_per_qubo, num_repeats=num_repeats)
    print('Solution:')
    print(solution)


def test_matplotlib():
    """Dummy test to make sure matplotlib works."""
    plt.plot([1, 2, 3])
    plt.show()


def plot_solution_error(points_per_qubo = 1):
    """Plots solution error as a function of number of grid points."""
    grid_from = 0
    grid_to = 1
    Ns = np.geomspace(10, 100, 5, dtype=int)
    y1 = 1
    qbits_integer = 2
    qbits_decimal = 30
    num_repeats = 200

    plot_data = np.empty((2, 2, len(Ns)))
    for i in range(len(Ns)):
        N = Ns[i]
        grid = np.linspace(grid_from, grid_to, N)
        f = np.exp(grid)
        dx = grid[1] - grid[0]

        H = 2 * qde.build_quadratic_minimization_matrix(len(f), dx)
        d = qde.build_quadratic_minimization_vector(f, dx, y1)
        solution_real = solve_qp(H, d)
        error = abs(solution_real[-1] - np.exp(grid_to))
        plot_data[0, :, i] = (N, error)

        solution_bin = qde.solve(f, dx, y1, qbits_integer, qbits_decimal, points_per_qubo, num_repeats=num_repeats)
        error = abs(solution_bin[-1] - np.exp(grid_to))
        plot_data[1, :, i] = (N, error)

    plt.plot(plot_data[0, 0, :], plot_data[0, 1, :], 'b.-', markersize=10)
    plt.plot(plot_data[1, 0, :], plot_data[1, 1, :], 'r.--', markersize=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.show()


if __name__ == '__main__':
    np.set_printoptions(precision=15, linewidth=200)
    plot_solution_error()
    #  test_exp_qp()
    #  test_exp_qubo()
