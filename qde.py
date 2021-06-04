"""This module contains functions that solve differential equations by transforming them to QUBO problems, which allows solution on quantum annealer.
"""
import dimod
import numpy as np
from dwave_qbsolv import QBSolv


def build_quadratic_minimization_matrix(npoints, dx):
    """Builds a matrix that defines quadratic minimization problem (H) corresponding to a given differential equation.

    Args:
        npoints (int): Number of discretization points for functions defining a given problem.
        dx (float): Grid step.

    Returns:
        numpy.ndarray (2D): Quadratic minimization matrix.
    """
    H = np.diag([2] * (npoints - 1)) + np.diag([-1] * (npoints - 2), 1) + np.diag([-1] * (npoints - 2), -1)
    H[-1, -1] = 1
    return H / dx ** 2


def build_quadratic_minimization_vector(f, dx, y1):
    """Builds a vector that defines quadratic minimization problem (d) corresponding to a given differential equation.

    Args:
        f (numpy.ndarray (1D)): Array of values of the derivative at the grid points.
        dx (float): Grid step.
        y1 (float): Solution's value at the leftmost point (boundary condition).

    Returns:
        numpy.ndarray (1D): Quadratic minimization vector.
    """
    d = -f[0:-1]
    d[0:-1] += f[1:-1]
    d[0] -= y1 / dx
    return d * 2 / dx


def build_discretization_matrix(qbits_integer, qbits_decimal):
    """Builds a discretization matrix (H~) for given number of qubits in integer and decimal parts.

    Args:
        qbits_integer (int): Number of qubits to represent integer part of each expansion coefficient (value) of the sample solution.
        qbits_decimal (int): Number of qubits to represent decimal part of each expansion coefficient.

    Returns:
        numpy.ndarray (2D): Discretization matrix.
    """
    j_range = range(-qbits_integer + 1, qbits_decimal + 1)
    return np.reshape([2 ** -(j1 + j2) for j1 in j_range for j2 in j_range], (len(j_range), len(j_range)))


def build_discretization_vector(qbits_integer, qbits_decimal):
    """Builds a discretization vector (d~) for given number of qubits in integer and decimal parts.

    Args:
        qbits_integer (int): Number of qubits to represent integer part of each expansion coefficient (value) of the sample solution.
        qbits_decimal (int): Number of qubits to represent decimal part of each expansion coefficient.

    Returns:
        numpy.ndarray (1D): Discretization vector.
    """
    j_range = range(-qbits_integer + 1, qbits_decimal + 1)
    return np.array([2 ** -j for j in j_range])


def build_qubo_matrix(f, dx, y1, qbits_integer, qbits_decimal):
    """Builds a QUBO matrix (Q) corresponding to a given differential equation. 

    A sample solution is represented by its values at grid points discretized using fixed point representation with set number of qubits.
    Derivative at each point of sample solution is calculated with a finite difference method and compared with the true derivative f.
    Sum of squares of their difference at all points of the sample solutions constitutes the target functional.

    Args:
        f (numpy.ndarray (1D)): Array of values of the derivative at the grid points.
        dx (float): Grid step.
        y1 (float): Solution's value at the leftmost point (boundary condition).
        qbits_integer (int): Number of qubits to represent integer part of each expansion coefficient (value) of the sample solution.
        qbits_decimal (int): Number of qubits to represent decimal part of each expansion coefficient.

    Returns:
        Q (numpy.ndarray): QUBO matrix.
        d_discret_elem (numpy.ndarray): Coefficient discretization vector.
    """
    H_cont = build_quadratic_minimization_matrix(len(f), dx)
    d_cont = build_quadratic_minimization_vector(f, dx, y1)
    H_discret_elem = build_discretization_matrix(qbits_integer, qbits_decimal)
    d_discret_elem = build_discretization_vector(qbits_integer, qbits_decimal)
    H_bin = np.block([[H_discret_elem * val for val in row] for row in H_cont])
    d_bin = np.block([d_discret_elem * val for val in d_cont])
    Q = H_bin + np.diag(d_bin)
    return Q, d_discret_elem


def solve(f, dx, y1, qbits_integer, qbits_decimal):
    """Solves a given differential equation, defined by f and y1, by formulating it as a QUBO problem with given discretization precision.

    Args:
        f (numpy.ndarray (1D)): Array of values of the derivative at the grid points.
        dx (float): Grid step.
        y1 (float): Solution's value at the leftmost point (boundary condition).
        qbits_integer (int): Number of qubits to represent integer part of each expansion coefficient (value) of the sample solution.
        qbits_decimal (int): Number of qubits to represent decimal part of each expansion coefficient.

    Returns:
        numpy.ndarray (1D): Values of solution function at grid points.
    """
    Q, d_discret_elem = build_qubo_matrix(f, dx, y1, qbits_integer, qbits_decimal)
    Q_dict = {(i, j) : Q[i, j] for i in range(Q.shape[0]) for j in range(Q.shape[1])}
    samples = QBSolv().sample_qubo(Q_dict)
    print(samples)
    ans_bin_dict = next(samples.samples())
    ans_bin = np.array([item[1] for item in ans_bin_dict.items()])
    ans_bin_2d = ans_bin.reshape(-1, len(d_discret_elem))
    ans_cont = np.sum(ans_bin_2d * d_discret_elem, 1)
    return ans_cont

