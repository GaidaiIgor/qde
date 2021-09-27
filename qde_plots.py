import addcopyfighandler
import findiff

from plots_general import my_plot
from test_core import *

addcopyfighandler.dummy_use = 1


def plot_potential_harmonic(**kwargs):
    equilibrium, force = (Hydrogen.equilibrium, Hydrogen.force_const)
    grid = np.linspace(-0.2, 0.2, 100) + equilibrium
    pot = force * (grid - equilibrium) ** 2
    axes = my_plot(grid, pot / Constants.eh_per_cm_1, **kwargs)
    axes.set_xlabel(r'$\mathrm{H-H\ dist, a_0}$')
    axes.set_ylabel(r'$\mathrm{Energy, cm^{-1}}$')
    return axes


def plot_potential_morse(**kwargs):
    grid = np.linspace(-0.7, 9, 1000) + r0
    pot = Hydrogen.get_potential_morse(grid)
    axes = my_plot(grid, pot / Constants.eh_per_cm_1, **kwargs)
    axes.set_xlabel(r'$\mathrm{r, a_0}$')
    axes.set_ylabel(r'$\mathrm{Energy, cm^{-1}}$')
    return axes


def plot_force_morse(**kwargs):
    grid = np.linspace(-0.7, 9, 1000) + re
    force = Hydrogen.get_force_morse(grid)
    axes = my_plot(grid, force / Constants.eh_per_cm_1, **kwargs)
    axes.set_xlabel(r'$\mathrm{r, a_0}$')
    axes.set_ylabel(r'$\mathrm{Force, cm^{-1} / a_0}$')
    axes.set_ylim(bottom=-20000, top=20000)
    return axes


def plot_solution_tr(t, r, **kwargs):
    axes = my_plot(t, r, **kwargs)
    axes.set_xlabel('t, a.u.')
    axes.set_ylabel('r, Bohr')
    return axes


def plot_solution_rp(r, p, **kwargs):
    axes = my_plot(r, p, **kwargs)
    axes.set_xlabel('r, Bohr')
    axes.set_ylabel('p, a.u.')
    return axes


def plot_solution_rp_tr(t, r, **kwargs):
    dt = t[1] - t[0]
    d_dt = findiff.FinDiff(0, dt)
    p = Hydrogen.mu * d_dt(r)
    return plot_solution_rp(r, p, **kwargs)


def plot_solution_rp_file(file_path='solution.txt', **kwargs):
    solution = np.loadtxt(file_path)
    return plot_solution_rp(solution[0, :], solution[1, :], **kwargs)


def plot_error(solution_n, true_solution_n, Ns, **kwargs):
    """Plots error at given values of Ns. solution_n and answer_n are function of n."""
    plot_data = np.empty((2, len(Ns)))
    for i in range(len(Ns)):
        N = Ns[i]
        solution = solution_n(N)
        true_solution = true_solution_n(N)
        # error = abs((solution[-1] - true_solution[-1]) / true_solution[-1]) * 100
        error = np.sqrt(sum((solution - true_solution) ** 2) / len(solution))
        # error = max(abs(solution - true_solution))
        plot_data[:, i] = (N, error)

    axes = my_plot(plot_data[0, :], plot_data[1, :], log=True, **kwargs)
    axes.set_xlabel('M')
    axes.set_ylabel('RMSE, Bohr')
    return axes


def plot_all_errors_vs_n_eq_1():
    """Plots errors as a function of N for multiple approaches, 1 equation per step."""
    Ns = np.geomspace(10, 1000, 5, dtype=int)
    analytical_solution_n = lambda n: get_analytical_solution(problem_id=0, N=n, time_max=400, initial_position=1.3)[1]

    solution_n = lambda n: np.loadtxt(f'../results/qp/eq_1/N_{n}/solution.txt')[0, :]
    axes = plot_error(solution_n, analytical_solution_n, Ns, label='QP')

    solution_n = lambda n: np.loadtxt(f'../results/qbsolv/eq_1/attempts_1/kd_15/N_{n}/solution.txt')[0, :]
    axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='QBSolv')

    solution_n = lambda n: np.loadtxt(f'../results/dwave/eq_1/attempts_1/N_{n}/solution.txt')[0, :]
    axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='DWave 1')

    # solution_n = lambda n: np.loadtxt(f'../results/dwave/eq_1/attempts_5/initial_1.3/N_{n}/solution.txt')[0, :]
    # axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='Dwave 5')

    solution_n = lambda n: np.loadtxt(f'../results/dwave/eq_1/attempts_10/N_{n}/solution.txt')[0, :]
    axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='Dwave 10')


def plot_all_errors_vs_n_eq_2():
    """Plots errors as a function of N for multiple approaches, 2 equations per step."""
    Ns = np.geomspace(10, 1000, 5, dtype=int)
    analytical_solution_n = lambda n: get_analytical_solution(problem_id=0, N=n, time_max=400, initial_position=1.3)[1]

    solution_n = lambda n: np.loadtxt(f'../results/qp/eq_2/N_{n}/solution.txt')[0, :]
    axes = plot_error(solution_n, analytical_solution_n, Ns, label='QP')

    solution_n = lambda n: np.loadtxt(f'../results/qbsolv/eq_2/attempts_1/N_{n}/solution.txt')[0, :]
    axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='QBSolv 1')

    # solution_n = lambda n: np.loadtxt(f'../results/qbsolv/eq_2/attempts_1/greedy/N_{n}/solution.txt')[0, :]
    # axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='QBSolv 1 greedy')

    solution_n = lambda n: np.loadtxt(f'../results/qbsolv/eq_2/attempts_1/scaled/N_{n}/solution.txt')[0, :]
    axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='QBSolv 1 scaled')

    solution_n = lambda n: np.loadtxt(f'../results/qbsolv/eq_2/attempts_10/N_{n}/solution.txt')[0, :]
    axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='QBSolv 10')

    solution_n = lambda n: np.loadtxt(f'../results/qbsolv/eq_2/attempts_10/scaled/N_{n}/solution.txt')[0, :]
    axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='QBSolv 10 scaled')

    # solution_n = lambda n: np.loadtxt(f'../results/dwave/eq_2/attempts_1/N_{n}/solution.txt')[0, :]
    # axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='DWave')

    # solution_n = lambda n: np.loadtxt(f'../results/dwave/eq_2/attempts_5/N_{n}/solution.txt')[0, :]
    # axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='DWave 5')

    solution_n = lambda n: np.loadtxt(f'../results/dwave/eq_2/attempts_10/at_20/N_{n}/solution.txt')[0, :]
    axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='DWave 10')

    # solution_n = lambda n: np.loadtxt(f'../results/dwave/eq_2/attempts_1/at_200/chain/N_{n}/solution.txt')[0, :]
    # axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='DWave+ 1')
    #
    # solution_n = lambda n: np.loadtxt(f'../results/dwave/eq_2/attempts_5/at_200/N_{n}/solution.txt')[0, :]
    # axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='DWave+ 5')

    solution_n = lambda n: np.loadtxt(f'../results/dwave/eq_2/attempts_1/greedy/N_{n}/solution.txt')[0, :]
    axes = plot_error(solution_n, analytical_solution_n, Ns, axes=axes, label='DWave 1 + greedy', color='m')


def plot_trajectories():
    """Plots a few representative trajectories."""
    grid, analytical_solution = get_analytical_solution(problem_id=0, N=1000, time_max=400, initial_position=1.3)
    axes = plot_solution_rp_tr(grid, analytical_solution, axes=None, marker='None', color='b', label='Analytical')
    axes = plot_solution_rp_file('../results/dwave/eq_1/attempts_1/N_1000/solution.txt', axes=axes, marker='None', color='g', label='DWave 1')
    axes = plot_solution_rp_file('../results/dwave/eq_1/attempts_10/N_1000/solution.txt', axes=axes, marker='None', color='k', label='DWave 10')
    axes = plot_solution_rp_file('../results/dwave/eq_2/attempts_1/greedy/N_1000/solution.txt', axes=axes, marker='None', color='m', label='DWave 1 + greedy')


def main():
    solution = get_solution(problem_id=0, N=50, time_max=400, initial_position=1.3, points_per_step=1, equations_per_step=2, max_attempts=1, max_error=1e-10, method='qp')[1]
    axes = plot_solution_rp(solution[0, :], solution[1, :], axes=None, marker='None')
    solution = get_solution(problem_id=0, N=50, time_max=400, initial_position=1.3, points_per_step=1, equations_per_step=2, max_attempts=1, max_error=1e-10, method='qbsolv')[1]
    axes = plot_solution_rp(solution[0, :], solution[1, :], axes=axes, marker='None')


if __name__ == '__main__':
    main()
