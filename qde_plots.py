import addcopyfighandler
import findiff
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from plots_general import my_plot, my_scatter
from test_core import Hydrogen, get_analytical_solution, get_qubo_solution

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
    axes.set_ylabel('r, a.u.')
    return axes


def plot_solution_rp(r, p, **kwargs):
    # axes = my_scatter(r, p, **kwargs)
    axes = my_plot(r, p, **kwargs)
    axes.set_xlabel('r, a.u.')
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


def plot_error(solution_n, true_answer_n, Ns=None, **kwargs):
    """Plots error at given values of Ns. solution_n and answer_n are function of n."""
    if Ns is None:
        Ns = np.geomspace(10, 100, 5, dtype=int)
    plot_data = np.empty((2, len(Ns)))
    for i in range(len(Ns)):
        N = Ns[i]
        solution = solution_n(N)
        true_answer = true_answer_n(N)
        error = abs((solution[-1] - true_ans[-1]) / true_ans[-1]) * 100
        plot_data[:, i] = (N, error)

    axes = my_plot(plot_data[0, :], plot_data[1, :], **kwargs)
    axes.set_xlabel('N')
    axes.set_ylabel('Error, %')
    return axes


def main():
    np.set_printoptions(precision=15, linewidth=200)
    mpl.rcParams['axes.prop_cycle'] = cycler(color='brgkcmy')

    grid, solution = get_analytical_solution(problem=22, N=1600, time_max=400)
    axes = plot_solution_rp_tr(grid, solution)

    grid, sln, errors = get_qp_solution(problem=22, N=50, time_max=400)
    axes = plot_solution_rp(sln[0, :], sln[1, :], axes=axes)

    # _, solution = get_qubo_solution(problem=21, N=200, time_max=400, sampler_name='qbsolv', num_repeats=100)
    # axes = plot_solution_rp(solution[0, :], solution[1, :], axes=axes)
    #
    # _, solution = get_qubo_solution(problem=21, N=200, time_max=400, sampler_name='dwave', num_reads=10000)
    # np.savetxt('solution.txt', solution)
    # axes = plot_solution_rp(solution[0, :], solution[1, :], axes=axes)

    if not mpl.is_interactive():
        plt.show()


if __name__ == '__main__':
    main()
