"""This module contains functions that solve differential equations by transforming them to QUBO problems, which allows solution on quantum annealer.
"""
import findiff
import numpy as np
import qpsolvers


def get_finite_difference_coefficients(deriv_order, accuracy_order):
    """Returns coefficients of a given forward finite difference scheme.

    Args:
        deriv_order (int): Order of derivative.
        accuracy_order (int): Order of accuracy.

    Returns:
        numpy.ndarray: 1D array of coefficients of selected scheme.
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

    Returns:
        energy_shift (float): Constant part of minimization functional.
    """
    energy_shift = funcs_i[0] ** 2
    first_unknown_point_global = known_points.shape[0]
    for deriv_ind in range(1, len(funcs_i)):
        deriv_order, accuracy_order, length, last_scheme_point_global = get_deriv_range(deriv_ind, point_ind, last_unknown_point_global, max_considered_accuracy)
        coeffs = get_finite_difference_coefficients(deriv_order, accuracy_order)
        func_factor = 2 * funcs_i[0] * funcs_i[deriv_ind] / dx ** deriv_order
        for scheme_point in range(length):
            c_factor = func_factor * coeffs[scheme_point]
            scheme_point_global = point_ind + scheme_point
            unknown_point = scheme_point_global - first_unknown_point_global
            if unknown_point < 0:
                energy_shift += c_factor * known_points[scheme_point_global]
            else:
                d[unknown_point] += c_factor

    return energy_shift


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

    Returns:
        energy_shift (float): Constant part of minimization functional.
    """
    energy_shift = 0
    first_unknown_point_global = known_points.shape[0]
    for deriv_ind1 in range(1, len(funcs_i)):
        deriv_order1, accuracy_order1, length1, last_scheme_point_global1 = get_deriv_range(deriv_ind1, point_ind, last_unknown_point_global, max_considered_accuracy)
        coeffs1 = get_finite_difference_coefficients(deriv_order1, accuracy_order1)
        for deriv_ind2 in range(1, len(funcs_i)):
            deriv_order2, accuracy_order2, length2, last_scheme_point_global2 = get_deriv_range(deriv_ind2, point_ind, last_unknown_point_global, max_considered_accuracy)
            coeffs2 = get_finite_difference_coefficients(deriv_order2, accuracy_order2)
            func_factor = funcs_i[deriv_ind1] * funcs_i[deriv_ind2] / dx ** (deriv_order1 + deriv_order2)
            for scheme_point1 in range(length1):
                scheme_point_global1 = point_ind + scheme_point1
                unknown_point1 = scheme_point_global1 - first_unknown_point_global
                for scheme_point2 in range(length2):
                    scheme_point_global2 = point_ind + scheme_point2
                    unknown_point2 = scheme_point_global2 - first_unknown_point_global
                    c_factor = func_factor * coeffs1[scheme_point1] * coeffs2[scheme_point2]
                    if unknown_point1 < 0 and unknown_point2 < 0:
                        energy_shift += c_factor * known_points[scheme_point_global1] * known_points[scheme_point_global2]
                    elif unknown_point1 >= 0 and unknown_point2 >= 0:
                        H[unknown_point1, unknown_point2] += c_factor
                    else:
                        unknown_ind = max(unknown_point1, unknown_point2)
                        known_ind = min(unknown_point1, unknown_point2) + first_unknown_point_global
                        d[unknown_ind] += c_factor * known_points[known_ind]

    return energy_shift


def build_qp_matrices(funcs, dx, known_points, max_considered_accuracy, points_per_step):
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
        energy_shift (float): Constant part of minimization functional.
    """
    first_unknown_point_global = known_points.shape[0]
    unknowns = min(points_per_step, funcs.shape[1] - first_unknown_point_global)
    last_unknown_point_global = first_unknown_point_global + unknowns - 1
    longest_scheme = funcs.shape[0] - 2 + max_considered_accuracy
    first_contributing_point = max(first_unknown_point_global - longest_scheme + 1, 0)
    last_contributing_point = max(last_unknown_point_global - longest_scheme + 1, 0)
    H = np.zeros((unknowns, unknowns))
    d = np.zeros(unknowns)
    energy_shift = 0
    for point_ind in range(first_contributing_point, last_contributing_point + 1):
        energy_shift += add_linear_terms_qp(d, point_ind, last_unknown_point_global, funcs[:, point_ind], dx, known_points, max_considered_accuracy)
        energy_shift += add_quadratic_terms_qp(H, d, point_ind, last_unknown_point_global, funcs[:, point_ind], dx, known_points, max_considered_accuracy)
    return 2*H, d, energy_shift


def solve_ode_qp(system_terms, grid, known_points, max_considered_accuracy, points_per_step, **kwargs):
    """Solves a given ODE system, defined by de_terms and known_points, by formulating it as a QP problem. Different equations are propagated one after another sequentially, not simultaneously.

    Args:
        system_terms (numpy.ndarray): 2D array of functions that define terms of a given system of differential equations. Rows - equations, columns - terms.
            Each term is a function that accepts x as first argument, and the values of all other functions in the order they are specified in system_terms as subsequent arguments.
        grid (numpy.ndarray): 1D Array of equidistant grid points.
        known_points (numpy.ndarray): 2D array of known points for each function in the system. Rows - equations, columns - points.
        max_considered_accuracy (int): Maximum accuracy order of finite difference scheme. Lower order is used if number of points is not sufficient.
        points_per_step (int): Number of points to vary in a single QP problem.
        kwargs (dict): args for QBSolv().sample_qubo.

    Returns:
        known_points (numpy.ndarray): 2D array with solution for all functions at all points of grid.
        errors (numpy.ndarray): 2D array with errors for each equation (rows) and each solved QP (columns).
    """
    known_points_extended = np.pad(known_points, [(0, 0), (0, len(grid) - known_points.shape[1])], constant_values=np.nan)
    funcs = np.array([[[term(*args) for args in np.vstack((grid, known_points_extended)).T] for term in equation_terms] for equation_terms in system_terms])
    dx = grid[1] - grid[0]
    errors = [[] for i in range(system_terms.shape[0])]
    while known_points.shape[1] < len(grid):
        all_solution_points_list = []
        for eq_ind in range(system_terms.shape[0]):
            H, d, energy_shift = build_qp_matrices(funcs[eq_ind, :, :], dx, known_points[eq_ind, :], max_considered_accuracy, points_per_step)
            solution_points = qpsolvers.solve_qp(H, d)
            error = np.dot(np.matmul(solution_points, H), solution_points) / 2 + np.dot(solution_points, d) + energy_shift
            all_solution_points_list.append(solution_points)
            errors[eq_ind].append(error)

        all_solution_points = np.array(all_solution_points_list)
        known_points = np.hstack((known_points, all_solution_points))
        # Update funcs
        update_cols = range(known_points.shape[1] - all_solution_points.shape[1], known_points.shape[1])
        funcs[:, :, update_cols] = [[[term(*args) for args in np.vstack((grid[update_cols], all_solution_points)).T] for term in equation_terms] for equation_terms in system_terms]

    return known_points, np.array(errors)


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
    energy_shift = funcs_i[0] ** 2
    bits_per_point = bits_integer + bits_decimal
    first_unknown_point_global = int(len(known_bits) / bits_per_point)
    for deriv_ind in range(1, len(funcs_i)):
        deriv_order, accuracy_order, scheme_length, last_scheme_point_global = get_deriv_range(deriv_ind, point_ind, last_unknown_point_global, max_considered_accuracy)
        coeffs = get_finite_difference_coefficients(deriv_order, accuracy_order)
        func_factor = 2 * funcs_i[0] * funcs_i[deriv_ind] / dx ** deriv_order
        for scheme_point in range(scheme_length):
            c_factor = func_factor * coeffs[scheme_point]
            scheme_point_global = point_ind + scheme_point
            if scheme_point_global < first_unknown_point_global:
                energy_shift += c_factor * bits_to_real(known_bits[scheme_point_global * bits_per_point : (scheme_point_global + 1) * bits_per_point], bits_integer)
            else:
                energy_shift -= c_factor * 2 ** (bits_integer - 1)
                unknown_point = scheme_point_global - first_unknown_point_global
                for unknown_bit in range(unknown_point * bits_per_point, (unknown_point + 1) * bits_per_point):
                    j = unknown_bit - unknown_point * bits_per_point - bits_integer + 1
                    d[unknown_bit] += c_factor * 2 ** (-j)


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


def build_qubo_matrix(funcs, dx, known_bits, bits_integer, bits_decimal, max_considered_accuracy, points_per_step):
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


def solve_ode_qubo(system_terms, grid, known_points, bits_integer, bits_decimal, max_considered_accuracy, points_per_step, sampler, max_attempts=1, restart_tolerance=0.05, **kwargs):
    """Solves a given differential equation, defined by funcs and known_points, by formulating it as a QUBO problem with given discretization precision.

    Args:
        system_terms (numpy.ndarray): 2D array of functions that define terms of a given system of differential equations. Rows - equations, columns - terms.
            Each term is a function that accepts x as first argument, and the values of all other functions in the order they are specified in system_terms as subsequent arguments.
        grid (numpy.ndarray): 1D Array of equidistant grid points.
        known_points (numpy.ndarray): 2D array of known points for each function in the system. Rows - equations, columns - points.
        bits_integer (int): Number of bits to represent integer part of each value of the sample solution.
        bits_decimal (int): Number of bits to represent decimal part of each value of the sample solution.
        max_considered_accuracy (int): Maximum accuracy order of finite difference scheme. Lower order is automatically used if number of points is not sufficient.
        points_per_step (int): Number of points to vary in the problem, defined by this matrix.
        sampler (dimod.core.Sampler): Sampler to use.
        max_attempts (int): Maximum number of times each QUBO can be solved (restarts can find a better solution).
        restart_tolerance (float): Allowed increase in energy relative to the predicted value.
        kwargs (dict): Sampler parameters.

    Returns:
        known_points (numpy.ndarray): 2D array with solution for all functions at all points of grid.
    """
    known_points = np.vectorize(lambda num: bits_to_real(real_to_bits(num, bits_integer, bits_decimal), bits_integer))(known_points)
    bits_per_point = bits_integer + bits_decimal
    known_bits = np.array([np.concatenate([real_to_bits(elem, bits_integer, bits_decimal) for elem in row]) for row in known_points])
    known_points_extended = np.pad(known_points, [(0, 0), (0, len(grid) - known_points.shape[1])], constant_values=np.nan)
    funcs = np.array([[[term(*args) for args in np.vstack((grid, known_points_extended)).T] for term in equation_terms] for equation_terms in system_terms])
    dx = grid[1] - grid[0]
    energies = [[] for i in range(system_terms.shape[0])]
    while known_points.shape[1] < len(grid):
        all_solution_bits_list = []
        all_solution_points_list = []
        for eq_ind in range(system_terms.shape[0]):
            lowest_energy = np.inf
            for attempt in range(max_attempts):
                Q = build_qubo_matrix(funcs[eq_ind, :, :], dx, known_bits[eq_ind, :], bits_integer, bits_decimal, max_considered_accuracy, points_per_step)
                job_label = f'Point {known_points.shape[1]}; Eq. {eq_ind}; Attempt {attempt + 1}'
                sample_set = sampler.sample_qubo(Q, label=job_label, **kwargs)
                samples_plain = np.array([list(sample.values()) for sample in sample_set])  # 2D, each row - solution (all bits together), sorted by energy
                solution_bits = samples_plain[0, :]
                solution_energy = sample_set.data_vectors['energy'][0]

                # if max_attempts > 0:


                all_solution_bits_list.append(solution_bits)
                solution_bits_shaped = np.reshape(solution_bits, (-1, bits_per_point))
                solution_points = np.array([bits_to_real(row_bits, bits_integer) for row_bits in solution_bits_shaped])
                all_solution_points_list.append(solution_points)
                energies[-1].append(solution_energy)

        all_solution_bits = np.array(all_solution_bits_list)
        all_solution_points = np.array(all_solution_points_list)
        known_bits = np.hstack((known_bits, all_solution_bits))
        known_points = np.hstack((known_points, all_solution_points))
        # Update funcs
        update_cols = range(known_points.shape[1] - all_solution_points.shape[1], known_points.shape[1])
        funcs[:, :, update_cols] = [[[term(*args) for args in np.vstack((grid[update_cols], all_solution_points)).T] for term in equation_terms] for equation_terms in system_terms]

    energies = np.array(energies).T
    return known_points, energies
