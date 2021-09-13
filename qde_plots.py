import addcopyfighandler
from cycler import cycler
import findiff
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from plots_general import my_plot, my_scatter
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


def plot_error(solution_n, true_solution, Ns=None, **kwargs):
    """Plots error at given values of Ns. solution_n and answer_n are function of n."""
    if Ns is None:
        Ns = np.geomspace(10, 100, 5, dtype=int)
    plot_data = np.empty((2, len(Ns)))
    for i in range(len(Ns)):
        N = Ns[i]
        solution = solution_n(N)
        error = abs((solution[-1] - true_solution[-1]) / true_solution[-1]) * 100
        plot_data[:, i] = (N, error)

    axes = my_plot(plot_data[0, :], plot_data[1, :], log=True, **kwargs)
    axes.set_xlabel('N')
    axes.set_ylabel('Error, %')
    return axes


def main():
    solution_n = lambda n: np.loadtxt(f'../results/qbsolv/N_{n}/solution.txt')[0, :]
    _, analytical_solution = get_analytical_solution(problem_id=0, N=1000, time_max=400, initial_position=1.3)
    Ns = np.geomspace(10, 1000, 5, dtype=int)
    axes = plot_error(solution_n, analytical_solution, Ns)

    if not mpl.is_interactive():
        plt.show()


if __name__ == '__main__':
    main()
