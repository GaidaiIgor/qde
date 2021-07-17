import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import qde
from qpsolvers import solve_qp

from plots_general import myplot

import addcopyfighandler


class Constants:
    a0_per_m = 1.889726124565062e+10
    me_per_amu = 1.822888484770040e+3
    eh_per_cm_1 = 4.556335256391438e-6
    eh_per_ev = 3.6749308136649e-2


class Hydrogen:
    """Hydrogen molecule (H2). All values are in atomic units."""
    equilibrium = 74e-12 * Constants.a0_per_m
    mu = 1.00782503207 * Constants.me_per_amu / 2  # Reduced mass
    freq = 4342 * Constants.eh_per_cm_1
    dissociation_energy = 4.52 * Constants.eh_per_ev
    force = mu * freq ** 2

    @staticmethod
    def get_harmonic_period():
        return np.pi * np.sqrt(2) / Hydrogen.freq

    @staticmethod
    def plot_potential_harmonic(**kwargs):
        equilibrium, force = (Hydrogen.equilibrium, Hydrogen.force)
        grid = np.linspace(-0.2, 0.2, 100) + equilibrium
        pot = force * (grid - equilibrium) ** 2
        axes = myplot(grid, pot / Constants.eh_per_cm_1, **kwargs)
        axes.set_xlabel(r'$\mathrm{H-H\ dist, a_0}$')
        axes.set_ylabel(r'$\mathrm{Energy, cm^{-1}}$')
        return axes

    @staticmethod
    def harmonic_trajectory(initial_position, initial_speed, t):
        w = Hydrogen.freq
        return Hydrogen.equilibrium + (initial_position - Hydrogen.equilibrium) * np.cos(np.sqrt(2) * w * t) + initial_speed / np.sqrt(2) / w * np.sin(np.sqrt(2) * w * t)

    @staticmethod
    def get_morse_a():
        return np.sqrt(Hydrogen.force / 2 / Hydrogen.dissociation_energy)

    @staticmethod
    def get_potential_morse(r):
        De = Hydrogen.dissociation_energy
        a = Hydrogen.get_morse_a()
        r0 = Hydrogen.equilibrium
        pot = De * (np.exp(-2 * a * (r - r0)) - 2 * np.exp(-a * (r - r0)))
        return pot

    @staticmethod
    def plot_potential_morse(**kwargs):
        grid = np.linspace(-0.7, 9, 1000) + r0
        pot = Hydrogen.get_potential_morse(grid)
        axes = myplot(grid, pot / Constants.eh_per_cm_1, **kwargs)
        axes.set_xlabel(r'$\mathrm{r, a_0}$')
        axes.set_ylabel(r'$\mathrm{Energy, cm^{-1}}$')
        return axes

    @staticmethod
    def get_force_morse(r):
        re = Hydrogen.equilibrium
        De = Hydrogen.dissociation_energy
        a = Hydrogen.get_morse_a()
        force = 2 * a * De * (np.exp(-2 * a * (r - re)) - np.exp(-a * (r - re)))
        return force

    @staticmethod
    def plot_force_morse(**kwargs):
        grid = np.linspace(-0.7, 9, 1000) + re
        force = Hydrogen.get_force_morse(grid)
        axes = myplot(grid, force / Constants.eh_per_cm_1, **kwargs)
        axes.set_xlabel(r'$\mathrm{r, a_0}$')
        axes.set_ylabel(r'$\mathrm{Force, cm^{-1} / a_0}$')
        axes.set_ylim(bottom=-20000, top=20000)
        return axes

    @staticmethod
    def morse_trajectory_v0(initial_position, t):
        """Returns morse trajectory at time t with specified initial position and 0 initial speed."""
        De = Hydrogen.dissociation_energy
        mu = Hydrogen.mu
        a = Hydrogen.get_morse_a()
        re = Hydrogen.equilibrium
        r0 = initial_position

        c1 = np.exp(a * re)
        c2 = np.exp(a * r0)
        c3 = -De * c1 / c2 * (2 - c1 / c2)
        c4 = De + c2 * c3 / c1
        tau = np.exp(np.sqrt(2 * c3 / mu, dtype=complex) * a * t)

        trajectory = np.log(c1 ** 2 * tau * (c3 * De + (De - c4 / tau) ** 2) / (2 * c1 * c3 * c4)) / a
        return trajectory


def get_problem(problem, **kwargs):
    """Returns problem-specific values: grid, de_terms, boundary condition and answer.
    kwargs: N, time_max, initial_position."""
    if problem == 0:
        # Problem: dy/dx = exp(x); y(0) = 1
        # Solution: y(x) = exp(x)
        N = kwargs.get('N', 11)
        grid = np.linspace(0, 1, N)
        de_terms = [None] * 3
        de_terms[0] = lambda x, y: -np.exp(x)
        de_terms[1] = lambda x, y: 0
        de_terms[2] = lambda x, y: 1
        known_points = np.exp(grid[0:1])
        solution = lambda x: np.exp(x)

    elif problem == 1:
        # Problem: r'' + 2 * w^2 * r - 2 * w^2 * re = 0; r(0) = r0; r'(0) = v0
        # Solution: r(t) = re + (r0 - re) * cos(2^0.5 * w * t) + v0 / 2^0.5 / w * sin(2^0.5 * w * t)
        N = kwargs.get('N', 10)
        w = Hydrogen.freq
        period = Hydrogen.get_harmonic_period()
        grid = np.linspace(0, period, N)
        de_terms = [None] * 4
        de_terms[0] = lambda x, y: -2 * w ** 2 * Hydrogen.equilibrium
        de_terms[1] = lambda x, y: 2 * w ** 2
        de_terms[2] = lambda x, y: 0
        de_terms[3] = lambda x, y: 1
        initial_position = 1.3
        initial_speed = 0
        known_points = np.array([initial_position, initial_position + initial_speed])
        solution = lambda t: Hydrogen.harmonic_trajectory(initial_position, initial_speed, t)

    elif problem == 2:
        # Problem: r'' = 2 * De * a / m * (exp(-2 * a * (r - re)) - exp(-a * (r - re))); r(0) = r0; r'(0) = 0
        # Solution: Hydrogen.morse_trajectory_v0
        time_max = kwargs.get('time_max', 1000)
        N = kwargs.get('N', 1001)
        initial_position = kwargs.get('initial_position', 1.3)
        grid = np.linspace(0, time_max, N)
        known_points = np.array([initial_position, initial_position])

        De = Hydrogen.dissociation_energy
        a = Hydrogen.get_morse_a()
        m = Hydrogen.mu
        re = Hydrogen.equilibrium
        de_terms = [None] * 4
        de_terms[0] = lambda t, r: -2 * De * a / m * (np.exp(-2 * a * (r - re)) - np.exp(-a * (r - re)))
        de_terms[1] = lambda t, r: 0
        de_terms[2] = lambda t, r: 0
        de_terms[3] = lambda t, r: 1
        solution = lambda t: Hydrogen.morse_trajectory_v0(initial_position, t)

    else:
        raise Exception('Unknown problem')

    return grid, de_terms, known_points, solution


def plot_analytical_solution(problem=0, **kwargs):
    """Plots analytical solution of a given problem"""
    grid, _, _, solution = get_problem(problem, **kwargs)
    solution_vals = solution(grid)
    if max(abs(np.imag(solution_vals))) < 1e-10:
        solution_vals = np.real(solution_vals)
    axes = myplot(grid, solution_vals, **kwargs)
    return axes


def get_qp_solution(problem, N=100, time_max=1000, initial_position=1.3, max_considered_accuracy=1, points_per_step=1):
    """Returns QP solution of a given problem."""
    grid, funcs, solution, true_solution = get_problem(problem, N=N, time_max=time_max, initial_position=initial_position)
    dx = grid[1] - grid[0]
    if points_per_step is None:
        points_per_step = len(grid)
    while len(solution) < len(grid):
        H, d = qde.build_qp_matrices_general(funcs, dx, solution, max_considered_accuracy, points_per_step)
        solution = np.concatenate((solution, solve_qp(2 * H, d)))
    return grid, solution, true_solution


def test_qp_solution(**kwargs):
    """Tests QP solution."""
    grid, solution, true_solution = get_qp_solution(**kwargs)
    print('Solution:')
    print(solution)
    ans = true_solution(grid[-1])
    error = abs((solution[-1] - ans) / ans) * 100
    print(f'Error: {error} %')


def plot_qp_solution(problem, N=100, time_max=1000, initial_position=1.3, max_considered_accuracy=1, points_per_step=1, **kwargs):
    """Plots QP solution of a given problem"""
    grid, solution, _ = get_qp_solution(problem, N, time_max, initial_position, max_considered_accuracy, points_per_step)
    axes = myplot(grid, solution, **kwargs)
    axes.set_xlabel('Time, a.u.')
    axes.set_ylabel('r, a.u.')
    return axes


def plot_qp_error(problem, time_max=1000, initial_position=1.3, max_considered_accuracy=1, points_per_step=1, **kwargs):
    """Plots QP solution error as a function of number of grid points."""
    Ns = np.geomspace(10, 100, 5, dtype=int)
    plot_data = np.empty((2, len(Ns)))
    for i in range(len(Ns)):
        N = Ns[i]
        grid, solution, true_solution = get_qp_solution(problem, N, time_max, initial_position, max_considered_accuracy, points_per_step)
        true_ans = true_solution(grid[-1])
        error = abs((solution[-1] - true_ans) / true_ans) * 100
        plot_data[:, i] = (N, error)

    axes = myplot(plot_data[0, :], plot_data[1, :], **kwargs)
    axes.set_xlabel('N')
    axes.set_ylabel('Error, %')
    return axes


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


def plot_qubo_solution(problem, N=100, time_max=300, initial_position=1.3, bits_integer=3, bits_decimal=15, max_considered_accuracy=1, points_per_step=1, **kwargs):
    """Plots QUBO solution of a given problem.
    kwargs: QBSolv().sample_qubo, myplot."""
    grid, de_terms, solution, _ = get_problem(problem, N=N, time_max=time_max, initial_position=initial_position)
    solution = qde.solve_general(de_terms, grid, solution, bits_integer, bits_decimal, max_considered_accuracy, points_per_step, **kwargs)

    axes = myplot(grid, solution, **kwargs)
    axes.set_xlabel('Time, a.u.')
    axes.set_ylabel('r, a.u.')
    return axes


def plot_qubo_error(problem, time_max=1000, initial_position=1.3, bits_integer=3, bits_decimal=15, max_considered_accuracy=1, points_per_step=1, **kwargs):
    """Plots QUBO error as a function of number of grid points."""
    Ns = np.geomspace(10, 100, 5, dtype=int)
    plot_data = np.empty((2, len(Ns)))
    for i in range(len(Ns)):
        N = Ns[i]
        grid, funcs, solution, true_solution = get_problem(problem, N=N, time_max=time_max, initial_position=initial_position)
        dx = grid[1] - grid[0]
        solution = qde.solve_general(funcs, dx, solution, bits_integer, bits_decimal, max_considered_accuracy, points_per_step, **kwargs)
        true_ans = true_solution(grid[-1])
        error = abs((solution[-1] - true_ans) / true_ans) * 100
        plot_data[:, i] = (N, error)

    axes = myplot(plot_data[0, :], plot_data[1, :], **kwargs)
    axes.set_xlabel('N')
    axes.set_ylabel('Error, %')
    return axes


if __name__ == '__main__':
    np.set_printoptions(precision=15, linewidth=200)
    plot_qubo_solution(problem=2, N=10)
    if not mpl.is_interactive():
        plt.show()
