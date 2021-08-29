import numpy as np
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave_qbsolv import QBSolv

import qde


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
    force_const = mu * freq ** 2

    @staticmethod
    def get_harmonic_period():
        return np.pi * np.sqrt(2) / Hydrogen.freq

    @staticmethod
    def get_force_harmonic(r):
        return -2 * Hydrogen.force_const * (r - Hydrogen.equilibrium)

    @staticmethod
    def harmonic_trajectory(initial_position, initial_speed, t):
        w = Hydrogen.freq
        return Hydrogen.equilibrium + (initial_position - Hydrogen.equilibrium) * np.cos(np.sqrt(2) * w * t) + initial_speed / np.sqrt(2) / w * np.sin(np.sqrt(2) * w * t)

    @staticmethod
    def get_morse_a():
        return np.sqrt(Hydrogen.force_const / 2 / Hydrogen.dissociation_energy)

    @staticmethod
    def get_potential_morse(r):
        De = Hydrogen.dissociation_energy
        a = Hydrogen.get_morse_a()
        r0 = Hydrogen.equilibrium
        pot = De * (np.exp(-2 * a * (r - r0)) - 2 * np.exp(-a * (r - r0)))
        return pot

    @staticmethod
    def get_force_morse(r):
        re = Hydrogen.equilibrium
        De = Hydrogen.dissociation_energy
        a = Hydrogen.get_morse_a()
        force = 2 * a * De * (np.exp(-2 * a * (r - re)) - np.exp(-a * (r - re)))
        return force

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
    """Returns problem-specific values: grid, system_terms, boundary condition and answer.
    kwargs: N, time_max, initial_position."""
    if problem == 0:
        # Problem: dy/dx = exp(x); y(0) = 1
        # Solution: y(x) = exp(x)
        N = kwargs.get('N', 11)
        grid = np.linspace(0, 1, N)
        system_terms = np.empty((1, 3), dtype=object)
        system_terms[0, 0] = lambda x, y: -np.exp(x)
        system_terms[0, 1] = lambda x, y: 0
        system_terms[0, 2] = lambda x, y: 1
        boundary_condition = np.exp(grid[0:1])
        solution = lambda x: np.exp(x)

    elif problem == 1:
        # Problem: r'' + 2 * w^2 * r - 2 * w^2 * re = 0; r(0) = r0; r'(0) = v0
        # Solution: r(t) = re + (r0 - re) * cos(2^0.5 * w * t) + v0 / 2^0.5 / w * sin(2^0.5 * w * t)
        N = kwargs.get('N', 10)
        w = Hydrogen.freq
        period = Hydrogen.get_harmonic_period()
        grid = np.linspace(0, period, N)
        system_terms = np.empty((1, 4), dtype=object)
        system_terms[0, 0] = lambda x, y: -2 * w ** 2 * Hydrogen.equilibrium
        system_terms[0, 1] = lambda x, y: 2 * w ** 2
        system_terms[0, 2] = lambda x, y: 0
        system_terms[0, 3] = lambda x, y: 1
        initial_position = 1.3
        initial_speed = 0
        boundary_condition = np.array([initial_position, initial_position + initial_speed])
        solution = lambda t: Hydrogen.harmonic_trajectory(initial_position, initial_speed, t)

    elif problem == 12:
        # Same as 1, but as a system of first-order equations, new version
        N = kwargs['N']
        initial_position = kwargs['initial_position']
        w = Hydrogen.freq
        period = Hydrogen.get_harmonic_period()
        grid = np.linspace(0, period, N)

        system_terms = np.empty(2, dtype=object)
        system_terms[0] = lambda t, r, p: p / Hydrogen.mu
        system_terms[1] = lambda t, r, p: Hydrogen.get_force_harmonic(r)

        boundary_condition = np.array([initial_position, 0])
        solution = lambda t: Hydrogen.harmonic_trajectory(initial_position, initial_speed, t)

    elif problem == 2:
        # Problem: r'' = 2 * De * a / m * (exp(-2 * a * (r - re)) - exp(-a * (r - re))); r(0) = r0; r'(0) = 0
        # Solution: Hydrogen.morse_trajectory_v0
        time_max = kwargs.get('time_max', 1000)
        N = kwargs.get('N', 1001)
        initial_position = kwargs.get('initial_position', 1.3)
        grid = np.linspace(0, time_max, N)
        boundary_condition = np.array([initial_position, initial_position])[np.newaxis, :]

        De = Hydrogen.dissociation_energy
        a = Hydrogen.get_morse_a()
        m = Hydrogen.mu
        re = Hydrogen.equilibrium

        system_terms = np.empty((1, 4), dtype=object)
        system_terms[0, 0] = lambda t, r: -2 * De * a / m * (np.exp(-2 * a * (r - re)) - np.exp(-a * (r - re)))
        system_terms[0, 1] = lambda t, r: 0
        system_terms[0, 2] = lambda t, r: 0
        system_terms[0, 3] = lambda t, r: 1

        solution = lambda t: Hydrogen.morse_trajectory_v0(initial_position, t)

    elif problem == 21:
        # Same as 2, but as a system of first-order equations
        time_max = kwargs.get('time_max', 1000)
        N = kwargs.get('N', 1001)
        initial_position = kwargs.get('initial_position', 1.3)
        grid = np.linspace(0, time_max, N)
        boundary_condition = np.array([initial_position, 0])[:, np.newaxis]

        system_terms = np.empty((2, 3), dtype=object)
        system_terms[0, 0] = lambda t, r, p: -p / Hydrogen.mu
        system_terms[1, 0] = lambda t, r, p: -Hydrogen.get_force_morse(r)
        system_terms[:, 1] = lambda t, r, p: 0
        system_terms[:, 2] = lambda t, r, p: 1

        solution = lambda t: Hydrogen.morse_trajectory_v0(initial_position, t)

    elif problem == 22:
        # Same as 2, but as a system of first-order equations, new version
        time_max = kwargs['time_max']
        N = kwargs['N']
        initial_position = kwargs['initial_position']
        grid = np.linspace(0, time_max, N)

        system_terms = np.empty(2, dtype=object)
        system_terms[0] = lambda t, r, p: p / Hydrogen.mu
        system_terms[1] = lambda t, r, p: Hydrogen.get_force_morse(r)

        boundary_condition = np.array([initial_position, 0])
        solution = lambda t: Hydrogen.morse_trajectory_v0(initial_position, t)

    else:
        raise Exception('Unknown problem')

    return grid, system_terms, boundary_condition, solution


def get_analytical_solution(problem=0, N=1000, time_max=300, initial_position=1.3, **kwargs):
    grid, _, _, solution = get_problem(problem, N=N, time_max=time_max, initial_position=initial_position, **kwargs)
    solution_vals = solution(grid)
    if max(abs(np.imag(solution_vals))) < 1e-10:
        solution_vals = np.real(solution_vals)
    return grid, solution_vals


def get_sampler(sampler_name, num_repeats):
    if sampler_name == 'qbsolv':
        return
    elif sampler_name == 'dwave':
        return qde.DWaveSamplerWrapper(num_repeats)
    else:
        raise Exception('Unknown sampler name')


def get_solver(solver_name, **kwargs):
    if solver_name == 'qp':
        return qde.QPSolver()
    else:
        if solver_name == 'qbsolv':
            sampler = qde.QBSolvWrapper(kwargs['num_repeats'])
        elif solver_name == 'dwave':
            sampler = qde.DWaveSamplerWrapper(kwargs['num_reads'])
        else:
            raise Exception('Unknown solver')
        return qde.QUBOSolver(kwargs['bits_integer'], kwargs['bits_decimal'], sampler)


def get_solution(problem, N=100, time_max=400, initial_position=1.3, points_per_step=1, equations_per_step=2, max_attempts=1, max_error=1e-10, solver_name='qp', num_repeats=100, num_reads=10000,
                 bits_integer=6, bits_decimal=15):
    grid, system_terms, boundary_condition, _ = get_problem(problem, N=N, time_max=time_max, initial_position=initial_position)
    solver = get_solver(solver_name, num_repeats=num_repeats, num_reads=num_reads, bits_integer=bits_integer, bits_decimal=bits_decimal)
    solution, errors = qde.solve_ode(system_terms, grid, boundary_condition, points_per_step, equations_per_step, solver, max_attempts, max_error)
    return grid, solution, errors


def main():
    # grid, sln, errors = get_qubo_solution(problem=21, N=50, time_max=400, sampler_name='qbsolv', num_repeats=100)

    _, solution, error = get_solution(problem=22, N=50, time_max=400, initial_position=1.1, points_per_step=1, equations_per_step=2, max_attempts=5, max_error=1e-10, solver_name='qp')
    np.savetxt('solution.txt', solution)
    # np.savetxt('error.txt', error)


if __name__ == '__main__':
    main()
