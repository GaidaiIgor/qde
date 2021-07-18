"""This module contains functions that solve differential equations by transforming them to QUBO problems, which allows solution on quantum annealer.
"""
import dimod
from dwave_qbsolv import QBSolv
import findiff
import numpy as np
import qpsolvers


def get_finite_difference_coefficients(deriv_order, accuracy_order):
    """Returns coefficients of a given forward finite difference scheme.

    Args:
        deriv_order (int): Order of derivative.
        accuracy_order (int): Order of accuracy.

    Returns:
        numpy.ndarray (1D): Coefficients of selected scheme.
    """
    assert accuracy_order >= 0, 'Not enough known points'

    if deriv_order == 0:
        return np.array([1])

    elif accuracy_order % 2 == 0:
        ans = findiff.coefficients(deriv_order, accuracy_order)['forward']
        return ans['coefficients']

    else:
        coeffs = None
        if accuracy_order == 1:
            if deriv_order == 1:
                coeffs = np.array([-1, 1])
            elif deriv_order == 2:
                coeffs = np.array([1, -2, 1])

        elif accuracy_order == 3:
            if deriv_order == 1:
                coeffs = np.array([-11/6, 3, -3/2, 1/3])
            elif deriv_order == 2:
                coeffs = np.array([35/12, -26/3, 19/2, -14/3, 11/12])

        elif accuracy_order == 5:
            if deriv_order == 1:
                coeffs = np.array([-137/60, 5, -5, 10/3, -5/4, 1/5])
            elif deriv_order == 2:
                coeffs = np.array([203/45, -87/5, 117/4, -254/9, 33/2, -27/5, 137/180])

        if coeffs is None:
            raise NotImplementedError('Not implemented combination of derivative and accuracy orders')
        else:
            return coeffs


def get_deriv_range(deriv_ind, point_ind, last_point_ind, max_considered_accuracy):
    """Returns derivative order, accuracy order, shift and first point index of a given derivative term.

    Args:
        deriv_ind (int): Index of term in DE equation.
        point_ind (int): Global index of point for which derivative range is requested.
        last_point_ind (int): Maximum global point index eligible to be included in a given scheme.
        max_considered_accuracy (int): Maximum considered accuracy order for a finite difference scheme.

    Returns:
        deriv_order (int): Derivative order of this term.
        selected_accuracy (int): Selected accuracy order for this term. If <1, then the remaining return values are invalid.
        length (int): Number of points in the selected scheme.
        last_scheme_ind_global (int): Global index of the last point in the selected scheme.
    """
    deriv_order = deriv_ind - 1
    if deriv_order == 0:
        selected_accuracy = 1
    else:
        max_possible_accuracy = last_point_ind - point_ind - deriv_order + 1
        selected_accuracy = min(max_considered_accuracy, max_possible_accuracy)

    length = deriv_order + selected_accuracy
    last_scheme_ind_global = point_ind + length - 1
    return deriv_order, selected_accuracy, length, last_scheme_ind_global


def add_linear_terms_qp(d, point_ind, last_unknown_point_global, funcs_i, dx, known_points, max_considered_accuracy):
    """Adds linear matrix elements resulting from linear terms of error functional for a given point.

    Args:
        d (numpy.ndarray (1D)): Current quadratic minimization vector to which linear matrix elements of specified point are added.
        point_ind (int): Global index of point for which the terms are to be calculated.
        last_unknown_point_global (int): Global index of the last unknown variable included in a given calculation.
        funcs_i (numpy.ndarray (2D)): Matrix with values of DE shift and multiplier functions. See `build_quadratic_minimization_matrices_general` for more details.
        dx (float): Grid step.
        known_points (numpy.ndarray (1D)): Array of known points in solution (continuous from the left end).
        max_considered_accuracy (int): Maximum accuracy order of finite difference scheme. Lower order is automatically used is number of points is not sufficient.
    """
    first_unknown_point_global = known_points.shape[0]
    for deriv_ind in reversed(range(1, len(funcs_i))):
        deriv_order, accuracy_order, length, last_scheme_point_global = get_deriv_range(deriv_ind, point_ind, last_unknown_point_global, max_considered_accuracy)
        if last_scheme_point_global < first_unknown_point_global:
            break
        coeffs = get_finite_difference_coefficients(deriv_order, accuracy_order)
        func_factor = 2 * funcs_i[0] * funcs_i[deriv_ind] / dx ** deriv_order
        for scheme_point in reversed(range(length)):
            unknown_point = point_ind + scheme_point - first_unknown_point_global
            if unknown_point < 0:
                break
            else:
                d[unknown_point] += func_factor * coeffs[scheme_point]


def add_quadratic_terms_qp(H, d, point_ind, last_unknown_point_global, funcs_i, dx, known_points, max_considered_accuracy):
    """Adds linear and quadratic matrix elements resulting from quadratic terms of error functional for a given point.

    Args:
        H (numpy.ndarray (2D)): Current quadratic minimization matrix to which quadratic matrix elements of specified point are added.
        d (numpy.ndarray (1D)): Current quadratic minimization vector to which linear matrix elements of specified point are added.
        point_ind (int): Global index of point for which the terms are to be calculated.
        last_unknown_point_global (int): Global index of the last unknown variable included in a given calculation.
        funcs_i (numpy.ndarray (2D)): Matrix with values of DE shift and multiplier functions. See `build_quadratic_minimization_matrices_general` for more details.
        dx (float): Grid step.
        known_points (numpy.ndarray (1D)): Array of known points in solution (continuous from the left end).
        max_considered_accuracy (int): Maximum accuracy order of finite difference scheme. Lower order is automatically used is number of points is not sufficient.
    """
    first_unknown_point_global = known_points.shape[0]
    for deriv_ind1 in reversed(range(1, len(funcs_i))):
        deriv_order1, accuracy_order1, length1, last_scheme_point_global1 = get_deriv_range(deriv_ind1, point_ind, last_unknown_point_global, max_considered_accuracy)
        coeffs1 = get_finite_difference_coefficients(deriv_order1, accuracy_order1)
        for deriv_ind2 in reversed(range(1, len(funcs_i))):
            deriv_order2, accuracy_order2, length2, last_scheme_point_global2 = get_deriv_range(deriv_ind2, point_ind, last_unknown_point_global, max_considered_accuracy)
            if last_scheme_point_global1 < first_unknown_point_global and last_scheme_point_global2 < first_unknown_point_global:
                break
            coeffs2 = get_finite_difference_coefficients(deriv_order2, accuracy_order2)
            func_factor = funcs_i[deriv_ind1] * funcs_i[deriv_ind2] / dx ** (deriv_order1 + deriv_order2)
            for scheme_point1 in reversed(range(length1)):
                unknown_point1 = point_ind + scheme_point1 - first_unknown_point_global
                for scheme_point2 in reversed(range(length2)):
                    unknown_point2 = point_ind + scheme_point2 - first_unknown_point_global
                    if unknown_point1 < 0 and unknown_point2 < 0:
                        break
                    else:
                        h_factor = func_factor * coeffs1[scheme_point1] * coeffs2[scheme_point2]
                        if unknown_point1 >= 0 and unknown_point2 >= 0:
                            H[unknown_point1, unknown_point2] += h_factor
                        else:
                            unknown_ind = max(unknown_point1, unknown_point2)
                            known_ind = min(unknown_point1, unknown_point2) + first_unknown_point_global
                            d[unknown_ind] += h_factor * known_points[known_ind]


def build_qp_matrices_general(funcs, dx, known_points, max_considered_accuracy, points_per_step):
    """Builds matrices H and d that define quadratic minimization problem corresponding to a given n-th order differential equation using up to k-th order difference schemes.

    Args:
        funcs (numpy.ndarray (2D)): Matrix with values of DE shift and multiplier functions. Functions are stored in rows. First row stores f (shift function).
            Subsequent i-th row stores f_(i-1) (multiplier function of (i-1)-th derivative term). Number of columns is equal to number of function discretization points.
        dx (float): Grid step.
        known_points (numpy.ndarray (1D)): Array of known points in solution (continuous from the left end).
        max_considered_accuracy (int): Maximum accuracy order of finite difference scheme. Lower order is automatically used is number of points is not sufficient.
        points_per_step (int): Number of points to vary in the problem, defined by this matrix.

    Returns:
        H (numpy.ndarray (2D)): Quadratic minimization matrix.
        d (numpy.ndarray (1D)): Quadratic minimization vector.
    """
    first_unknown_point_global = known_points.shape[0]
    unknowns = min(points_per_step, funcs.shape[1] - first_unknown_point_global)
    last_unknown_point_global = first_unknown_point_global + unknowns - 1
    longest_scheme = funcs.shape[0] - 2 + max_considered_accuracy
    first_contributing_point = max(first_unknown_point_global - longest_scheme + 1, 0)
    last_contributing_point = max(last_unknown_point_global - longest_scheme + 1, 0)
    H = np.zeros((unknowns, unknowns))
    d = np.zeros(unknowns)
    for point_ind in range(first_contributing_point, last_contributing_point + 1):
        add_linear_terms_qp(d, point_ind, last_unknown_point_global, funcs[:, point_ind], dx, known_points, max_considered_accuracy)
        add_quadratic_terms_qp(H, d, point_ind, last_unknown_point_global, funcs[:, point_ind], dx, known_points, max_considered_accuracy)
    return H, d


def solve_qp_general(de_terms, grid, known_points, max_considered_accuracy, points_per_step, **kwargs):
    """Solves a given differential equation, defined by funcs and known_points, by formulating it as a QUBO problem with given discretization precision.

    Args:
        de_terms (List[f(x, y)]): List of functions that define terms of a given DE.
        grid (numpy.ndarray (1D)): Array of equidistant grid points (x).
        known_points (numpy.ndarray (1D)): Array of known points (y).
        max_considered_accuracy (int): Maximum accuracy order of finite difference scheme. Lower order is automatically used if number of points is not sufficient.
        points_per_step (int): Number of points to vary in the problem, defined by this matrix.
        kwargs (dict): args for QBSolv().sample_qubo.

    Returns:
        known_points (numpy.ndarray (1D)): Solution at all points of grid.
    """
    if points_per_step is None:
        points_per_step = len(grid)
    known_points_extended = np.pad(known_points, (0, len(grid) - len(known_points)), constant_values=np.nan)
    funcs = np.array([[term(*args) for args in zip(grid, known_points_extended)] for term in de_terms])
    dx = grid[1] - grid[0]
    while len(known_points) < len(grid):
        H, d = build_qp_matrices_general(funcs, dx, known_points, max_considered_accuracy, points_per_step)
        solution_points = qpsolvers.solve_qp(2 * H, d)
        known_points = np.concatenate((known_points, solution_points))
        # Update funcs
        update_cols = range(len(known_points)-len(solution_points), len(known_points))
        funcs[:, update_cols] = [[term(*args) for args in zip(grid[update_cols], solution_points)] for term in de_terms]
    return known_points


def real_to_bits(num, bits_integer, bits_decimal):
    """Returns the closest binary representation of a given real number.

    Args:
        num (float): Number to convert.
        bits_integer (int): Number of bits to represent integer part of number.
        bits_decimal (int): Number of bits to represent decimal part of number.

    Returns:
        bits (numpy.ndarray (1D)): Array of bits.
    """
    bits = np.zeros(bits_integer + bits_decimal, dtype=int)
    represented = -2 ** (bits_integer - 1)
    for i in range(len(bits)):
        bit_value = 2 ** (bits_integer - 1 - i)
        if represented + bit_value <= num:
            bits[i] = 1
            represented += bit_value
    return bits


def bits_to_real(bits, bits_integer):
    """Returns a real number represented by given binary representation.

    Args:
        bits (numpy.ndarray (1D)): Array of bits.
        bits_integer (int): Number of bits to represent integer part of number.

    Returns:
        num: Represented real number.
    """
    bits_decimal = len(bits) - bits_integer
    discretization_vector = [2 ** -j for j in range(-bits_integer + 1, bits_decimal + 1)]
    return np.dot(bits, discretization_vector) - 2 ** (bits_integer - 1)


def add_linear_terms_qubo(d, point_ind, last_unknown_point_global, funcs_i, dx, known_bits, bits_integer, bits_decimal, max_considered_accuracy):
    """Adds linear matrix elements resulting from linear terms of error functional for a given point.

    Args:
        d (numpy.ndarray (1D)): Current quadratic minimization vector to which linear matrix elements of specified point are added.
        point_ind (int): Global index of point for which the terms are to be calculated.
        last_unknown_point_global (int): Global index of the last unknown variable included in a given calculation.
        funcs_i (numpy.ndarray (1D)): Values of all terms defining DE at the current point.
        dx (float): Grid step.
        known_bits (numpy.ndarray (1D)): Array of known bits in solution (continuous from the left end).
        bits_integer (int): Number of bits to represent integer part of coefficients.
        bits_decimal (int): Number of bits to represent decimal part of coefficients.
        max_considered_accuracy (int): Maximum accuracy order of finite difference scheme. Lower order is automatically used is number of points is not sufficient.
    """
    bits_per_point = bits_integer + bits_decimal
    first_unknown_point_global = int(len(known_bits) / bits_per_point)
    for deriv_ind in reversed(range(1, len(funcs_i))):
        deriv_order, accuracy_order, scheme_length, last_scheme_point_global = get_deriv_range(deriv_ind, point_ind, last_unknown_point_global, max_considered_accuracy)
        if last_scheme_point_global < first_unknown_point_global:
            break
        coeffs = get_finite_difference_coefficients(deriv_order, accuracy_order)
        func_factor = 2 * funcs_i[0] * funcs_i[deriv_ind] / dx ** deriv_order
        for scheme_point in reversed(range(scheme_length)):
            unknown_point = point_ind + scheme_point - first_unknown_point_global
            if unknown_point < 0:
                break
            else:
                for unknown_bit in range(unknown_point * bits_per_point, (unknown_point + 1) * bits_per_point):
                    j = unknown_bit - unknown_point * bits_per_point - bits_integer + 1
                    d[unknown_bit] += func_factor * coeffs[scheme_point] * 2 ** (-j)


def add_quadratic_terms_qubo(H, d, point_ind, last_unknown_point_global, funcs_i, dx, known_bits, bits_integer, bits_decimal, max_considered_accuracy):
    """Adds linear and quadratic matrix elements resulting from quadratic terms of error functional for a given point.

    Args:
        H (numpy.ndarray (2D)): Current quadratic minimization matrix to which quadratic matrix elements of specified point are added.
        d (numpy.ndarray (1D)): Current quadratic minimization vector to which linear matrix elements of specified point are added.
        point_ind (int): Global index of point for which the terms are to be calculated.
        last_unknown_point_global (int): Global index of the last unknown variable included in a given calculation.
        funcs_i (numpy.ndarray (1D)): Values of all terms defining DE at the current point.
        dx (float): Grid step.
        known_bits (numpy.ndarray (1D)): Array of known bits in solution (continuous from the left end).
        bits_integer (int): Number of bits to represent integer part of coefficients.
        bits_decimal (int): Number of bits to represent decimal part of coefficients.
        max_considered_accuracy (int): Maximum accuracy order of finite difference scheme. Lower order is automatically used is number of points is not sufficient.
    """
    bits_per_point = bits_integer + bits_decimal
    first_unknown_point_global = int(len(known_bits) / bits_per_point)
    for deriv_ind1 in reversed(range(1, len(funcs_i))):
        deriv_order1, accuracy_order1, scheme_length1, last_scheme_point_global1 = get_deriv_range(deriv_ind1, point_ind, last_unknown_point_global, max_considered_accuracy)
        coeffs1 = get_finite_difference_coefficients(deriv_order1, accuracy_order1)
        for deriv_ind2 in reversed(range(1, len(funcs_i))):
            deriv_order2, accuracy_order2, scheme_length2, last_scheme_point_global2 = get_deriv_range(deriv_ind2, point_ind, last_unknown_point_global, max_considered_accuracy)
            if last_scheme_point_global1 < first_unknown_point_global and last_scheme_point_global2 < first_unknown_point_global:
                break
            coeffs2 = get_finite_difference_coefficients(deriv_order2, accuracy_order2)
            func_factor = funcs_i[deriv_ind1] * funcs_i[deriv_ind2] / dx ** (deriv_order1 + deriv_order2)
            for scheme_point1 in reversed(range(scheme_length1)):
                unknown_point1 = point_ind + scheme_point1 - first_unknown_point_global
                for scheme_point2 in reversed(range(scheme_length2)):
                    unknown_point2 = point_ind + scheme_point2 - first_unknown_point_global
                    if unknown_point1 < 0 and unknown_point2 < 0:
                        break
                    else:
                        c_factor = func_factor * coeffs1[scheme_point1] * coeffs2[scheme_point2]
                        if unknown_point1 >= 0:
                            for unknown_bit in range(unknown_point1 * bits_per_point, (unknown_point1 + 1) * bits_per_point):
                                j = unknown_bit - unknown_point1 * bits_per_point - bits_integer + 1
                                d[unknown_bit] -= c_factor * 2 ** (bits_integer - 1 - j)

                        if unknown_point2 >= 0:
                            for unknown_bit in range(unknown_point2 * bits_per_point, (unknown_point2 + 1) * bits_per_point):
                                j = unknown_bit - unknown_point2 * bits_per_point - bits_integer + 1
                                d[unknown_bit] -= c_factor * 2 ** (bits_integer - 1 - j)

                        if unknown_point1 >= 0 and unknown_point2 >= 0:
                            for unknown_bit1 in range(unknown_point1 * bits_per_point, (unknown_point1 + 1) * bits_per_point):
                                j1 = unknown_bit1 - unknown_point1 * bits_per_point - bits_integer + 1
                                for unknown_bit2 in range(unknown_point2 * bits_per_point, (unknown_point2 + 1) * bits_per_point):
                                    j2 = unknown_bit2 - unknown_point2 * bits_per_point - bits_integer + 1
                                    H[unknown_bit1, unknown_bit2] += c_factor * 2 ** -(j1 + j2)
                        else:
                            unknown_point = max(unknown_point1, unknown_point2)
                            known_point_global = min(unknown_point1, unknown_point2) + first_unknown_point_global
                            for unknown_bit in range(unknown_point * bits_per_point, (unknown_point + 1) * bits_per_point):
                                j1 = unknown_bit - unknown_point * bits_per_point - bits_integer + 1
                                for known_bit in range(known_point_global * bits_per_point, (known_point_global + 1) * bits_per_point):
                                    j2 = known_bit - known_point_global * bits_per_point - bits_integer + 1
                                    d[unknown_bit] += c_factor * known_bits[known_bit] * 2 ** -(j1 + j2)


def build_qubo_matrix_general(funcs, dx, known_bits, bits_integer, bits_decimal, max_considered_accuracy, points_per_step):
    """Builds matrix Q that defines quadratic unconstrained binary optimization (QUBO) problem corresponding to a given n-th order differential equation using up to k-th order difference schemes.

    Args:
        funcs (numpy.ndarray (2D)): Matrix with values of DE shift and multiplier functions. Functions are stored in rows. First row stores f (shift function).
            Subsequent i-th row stores f_(i-1) (multiplier function of (i-1)-th derivative term). Number of columns is equal to number of function discretization points.
        dx (float): Grid step.
        known_bits (numpy.ndarray (1D)): Array of known bits in solution (continuous from the left end).
        bits_integer (int): Number of bits to represent integer part of coefficients.
        bits_decimal (int): Number of bits to represent decimal part of coefficients.
        max_considered_accuracy (int): Maximum accuracy order of finite difference scheme. Lower order is automatically used if number of points is not sufficient.
        points_per_step (int): Number of points to vary in the problem, defined by this matrix.

    Returns:
        Q (numpy.ndarray (2D)): QUBO matrix.
    """
    bits_per_point = bits_integer + bits_decimal
    first_unknown_point_global = int(len(known_bits) / bits_per_point)
    unknown_points = min(points_per_step, funcs.shape[1] - first_unknown_point_global)
    unknown_bits = unknown_points * bits_per_point
    last_unknown_point_global = first_unknown_point_global + unknown_points - 1
    longest_scheme = funcs.shape[0] - 2 + max_considered_accuracy
    first_contributing_point = max(first_unknown_point_global - longest_scheme + 1, 0)
    last_contributing_point = max(last_unknown_point_global - longest_scheme + 1, 0)
    H = np.zeros((unknown_bits, unknown_bits))
    d = np.zeros(unknown_bits)
    for point_ind in range(first_contributing_point, last_contributing_point + 1):
        add_linear_terms_qubo(d, point_ind, last_unknown_point_global, funcs[:, point_ind], dx, known_bits, bits_integer, bits_decimal, max_considered_accuracy)
        add_quadratic_terms_qubo(H, d, point_ind, last_unknown_point_global, funcs[:, point_ind], dx, known_bits, bits_integer, bits_decimal, max_considered_accuracy)
    Q = H + np.diag(d)
    return Q


def solve_qubo_general(de_terms, grid, known_points, bits_integer, bits_decimal, max_considered_accuracy, points_per_step, **kwargs):
    """Solves a given differential equation, defined by funcs and known_points, by formulating it as a QUBO problem with given discretization precision.

    Args:
        de_terms (List[f(x, y)]): List of functions that define terms of a given DE.
        grid (numpy.ndarray (1D)): Array of equidistant grid points (x).
        known_points (numpy.ndarray (1D)): Array of known points (y).
        bits_integer (int): Number of bits to represent integer part of each value of the sample solution.
        bits_decimal (int): Number of bits to represent decimal part of each value of the sample solution.
        max_considered_accuracy (int): Maximum accuracy order of finite difference scheme. Lower order is automatically used if number of points is not sufficient.
        points_per_step (int): Number of points to vary in the problem, defined by this matrix.
        kwargs (dict): args for QBSolv().sample_qubo.

    Returns:
        known_points (numpy.ndarray (1D)): Solution at all points of grid.
    """
    bits_per_point = bits_integer + bits_decimal
    known_bits = np.concatenate(list(map(lambda x: real_to_bits(x, bits_integer, bits_decimal), known_points)))
    known_points_extended = np.pad(known_points, (0, len(grid) - len(known_points)), constant_values=np.nan)
    funcs = np.array([[term(*args) for args in zip(grid, known_points_extended)] for term in de_terms])
    dx = grid[1] - grid[0]
    while len(known_bits) < len(grid) * bits_per_point:
        Q = build_qubo_matrix_general(funcs, dx, known_bits, bits_integer, bits_decimal, max_considered_accuracy, points_per_step)
        # sample_set = dimod.ExactSolver().sample_qubo(Q)
        sample_set = QBSolv().sample_qubo(Q, **kwargs)
        samples_plain = np.array([list(sample.values()) for sample in sample_set])  # 2D, each row - solution (all bits together), sorted by energy
        solution_bits = samples_plain[0, :]  # Take best sample
        known_bits = np.concatenate((known_bits, solution_bits))
        solution_bits_shaped = np.reshape(solution_bits, (-1, bits_per_point))
        solution_points = np.apply_along_axis(lambda point_bits: bits_to_real(point_bits, bits_integer), 1, solution_bits_shaped)
        known_points = np.concatenate((known_points, solution_points))
        # Update funcs
        update_cols = range(len(known_points)-len(solution_points), len(known_points))
        funcs[:, update_cols] = [[term(*args) for args in zip(grid[update_cols], solution_points)] for term in de_terms]

    return known_points


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


def build_quadratic_minimization_matrix(npoints, dx):
    """Builds a matrix that defines quadratic minimization problem (H) corresponding to a given 1st order differential equation using 1st order forward difference scheme.

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


def build_qubo_matrix(f, dx, y1, H_discret_elem, d_discret_elem, signed):
    """Builds a QUBO matrix (Q) corresponding to a given differential equation. 

    A sample solution is represented by its values at grid points discretized using fixed point representation with set number of qubits.
    Derivative at each point of sample solution is calculated with a finite difference method and compared with the true derivative f.
    Sum of squares of their difference at all points of the sample solutions constitutes the target functional.

    Args:
        f (numpy.ndarray (1D)): Array of values of the derivative at the grid points.
        dx (float): Grid step.
        y1 (float): Solution's value at the leftmost point (boundary condition).
        H_discret_elem (numpy.ndarray (2D)): Matrix discretization element for given number of qubits (see also build_discretization_matrix).
        d_discret_elem (numpy.ndarray (1D)): Vector discretization element for given number of qubits (see also build_discretization_vector).
        signed (bool): Whether to use signed or unsigned number representation. Using signed numbers shifts representation range from 0..L to -L/2..L/2, where L = 2 ** qbits_integer.

    Returns:
        Q (numpy.ndarray): QUBO matrix.
    """
    H_cont = build_quadratic_minimization_matrix(len(f), dx)
    d_cont = build_quadratic_minimization_vector(f, dx, y1)
    if signed:
        d_cont[0] -= 2 * d_discret_elem[0] / dx ** 2

    H_bin = np.block([[H_discret_elem * val for val in row] for row in H_cont])
    d_bin = np.block([d_discret_elem * val for val in d_cont])
    Q = H_bin + np.diag(d_bin)
    return Q


def solve(f, dx, y1, qbits_integer, qbits_decimal, signed=True, points_per_qubo=1, average_solutions=False, **kwargs):
    """Solves a given differential equation, defined by f and y1, by formulating it as a QUBO problem with given discretization precision.

    Args:
        f (numpy.ndarray (1D)): Array of values of the derivative at the grid points.
        dx (float): Grid step.
        y1 (float): Solution's value at the leftmost point (boundary condition).
        qbits_integer (int): Number of qubits to represent integer part of each expansion coefficient (value) of the sample solution.
        qbits_decimal (int): Number of qubits to represent decimal part of each expansion coefficient.
        signed (bool): Whether to use signed or unsigned number representation. Using signed numbers shifts representation range from 0..L to -L/2..L/2, where L = 2 ** qbits_integer.
        points_per_qubo (int): Number of points to propagate in each QUBO. Last point from the previous solution is used as boundary condition for the next solution.
        average_solutions (bool): If true, all found solutions are averaged according to number of times they were found. If false, only the best solution is considered.
        kwargs (dict): Additional keyword arguments to QBSolv().sample_qubo

    Returns:
        numpy.ndarray (2D): Values of the best found solution function at grid points.
    """
    solution = np.empty(len(f))
    solution[0] = y1

    H_discret_elem = build_discretization_matrix(qbits_integer, qbits_decimal)
    d_discret_elem = build_discretization_vector(qbits_integer, qbits_decimal)
    error = 0
    for i in range(1, len(f), points_per_qubo):
        y1 = solution[i - 1]
        next_i = min(i + points_per_qubo, len(f))
        next_f = f[i - 1 : next_i]
        Q = build_qubo_matrix(next_f, dx, y1, H_discret_elem, d_discret_elem, signed)
        sample_set = QBSolv().sample_qubo(Q, **kwargs)
        samples_bin = np.array([list(sample.values()) for sample in sample_set])
        samples_bin_structured = samples_bin.reshape((samples_bin.shape[0], -1, len(d_discret_elem)))
        samples_cont = np.sum(samples_bin_structured * d_discret_elem, 2)
        error_shift = (y1 ** 2 + 2 * y1 * next_f[0] * dx) / dx ** 2 + np.sum(next_f[0:-1] ** 2)
        if signed:
            samples_cont -= 2 ** (qbits_integer - 1)
            error_shift += (4 ** (qbits_integer - 1) + 2 ** qbits_integer * (y1 + next_f[0] * dx)) / dx ** 2

        if average_solutions:
            num_occurrences = np.array([sample.num_occurrences for sample in sample_set.data()])
            weights = num_occurrences / np.sum(num_occurrences)
            solution[i:next_i] = np.sum(samples_cont * weights[:, np.newaxis], 0)
            all_energies = np.array([sample.energy for sample in sample_set.data()])
            solution_energy = np.sum(all_energies * weights)
        else:
            solution[i:next_i] = samples_cont[0, :]
            solution_energy = next(sample_set.data()).energy

        error += solution_energy + error_shift

    return solution, error
