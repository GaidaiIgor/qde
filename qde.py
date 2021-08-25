"""This module contains functions that solve differential equations by transforming them to QUBO problems, which allows solution on quantum annealer.
"""
import numpy as np
import qpsolvers


def add_symmetric(H, ind1, ind2, value):
    """Splits specified value between the two off-diagonals of H.

    Args:
        H (numpy.ndarray): Matrix to which value is added.
        ind1 (int): first index of position in H where value is added.
        ind2 (int): second index of position in H where value is added.
        value (float): value to add.
    """
    H[ind1, ind2] += value / 2
    H[ind2, ind1] += value / 2


def add_point_terms_qp(H, d, point_ind, funcs_i, dx, known_points=None):
    """Adds functional terms for a given point to H and d.

    Args:
        H (numpy.ndarray): Current quadratic minimization matrix to which quadratic terms of specified point are added.
        d (numpy.ndarray): Current quadratic minimization vector to which linear terms of specified point are added.
        point_ind (int): Local point index within the current job.
        funcs_i (numpy.ndarray): 2D array with values of approximated rhs terms at the current point. Equations are along rows, terms along columns.
        dx (float): Grid step.
        known_points (numpy.ndarray): When adding terms for the last known point, this is 1D array of the values of each function at that point, otherwise None.

    Returns:
        energy_shift (float): Constant part of minimization functional.
    """
    energy_shift = 0
    get_unknown_ind = lambda point, eq: (point - 1) * funcs_i.shape[0] + eq
    for eq_ind in range(funcs_i.shape[0]):
        next_unknown_ind = get_unknown_ind(point_ind + 1, eq_ind)
        H[next_unknown_ind, next_unknown_ind] += 1 / dx ** 2
        d[next_unknown_ind] += -2 * funcs_i[eq_ind, 0] / dx
        energy_shift += funcs_i[eq_ind, 0] ** 2

        if point_ind == 0:
            # Current point is known
            assert known_points is not None, 'known_points have to be supplied for 0th point in each job'
            d[next_unknown_ind] += -2 * known_points[eq_ind] / dx ** 2
            energy_shift += (known_points[eq_ind] / dx) ** 2
            energy_shift += 2 * known_points[eq_ind] * funcs_i[eq_ind, 0] / dx
            for term_ind in range(1, funcs_i.shape[1]):
                d[next_unknown_ind] += -2 * funcs_i[eq_ind, term_ind] * known_points[term_ind - 1] / dx
                energy_shift += 2 * funcs_i[eq_ind, term_ind] * known_points[term_ind - 1] * known_points[eq_ind] / dx
                energy_shift += 2 * funcs_i[eq_ind, 0] * funcs_i[eq_ind, term_ind] * known_points[term_ind - 1]
                for term_ind2 in range(1, funcs_i.shape[1]):
                    energy_shift += funcs_i[eq_ind, term_ind] * funcs_i[eq_ind, term_ind2] * known_points[term_ind - 1] * known_points[term_ind2 - 1]
        else:
            unknown_ind = get_unknown_ind(point_ind, eq_ind)
            add_symmetric(H, unknown_ind, next_unknown_ind, -2 / dx ** 2)
            H[unknown_ind, unknown_ind] += 1 / dx ** 2
            d[unknown_ind] += 2 * funcs_i[eq_ind, 0] / dx
            for term_ind in range(1, funcs_i.shape[1]):
                term_unknown_ind = get_unknown_ind(point_ind, term_ind - 1)
                add_symmetric(H, term_unknown_ind, next_unknown_ind, -2 * funcs_i[eq_ind, term_ind] / dx)
                add_symmetric(H, term_unknown_ind, unknown_ind, 2 * funcs_i[eq_ind, term_ind] / dx)
                d[term_unknown_ind] += 2 * funcs_i[eq_ind, 0] * funcs_i[eq_ind, term_ind]
                for term_ind2 in range(1, funcs_i.shape[1]):
                    term_unknown_ind2 = get_unknown_ind(point_ind, term_ind2 - 1)
                    add_symmetric(H, term_unknown_ind, term_unknown_ind2, funcs_i[eq_ind, term_ind] * funcs_i[eq_ind, term_ind2])

    return energy_shift


def build_qp_matrices(funcs, dx, known_points):
    """Builds matrices H and d that define quadratic minimization problem corresponding to a given system of differential equations.

    Args:
        funcs (numpy.ndarray): 3D array with values of approximated rhs terms at all points of this job. 1st dim - equations, 2nd dim - terms, 3rd dim - points.
        dx (float): Grid step.
        known_points (numpy.ndarray): 1D array of known points for each function at 0th point (boundary condition).

    Returns:
        H (numpy.ndarray (2D)): Quadratic minimization matrix.
        d (numpy.ndarray (1D)): Quadratic minimization vector.
        energy_shift (float): Constant part of minimization functional.
    """
    unknowns = funcs.shape[0] * funcs.shape[2]
    H = np.zeros((unknowns, unknowns))
    d = np.zeros(unknowns)
    energy_shift = 0
    for point_ind in range(funcs.shape[2]):
        energy_shift += add_point_terms_qp(H, d, point_ind, funcs[:, :, point_ind], dx, known_points)
    return 2*H, d, energy_shift


def calculate_term_coefficients(system_terms, approximation_point, sampling_steps, grid):
    """Linearly approximates system rhs in the vicinity of a given point.

    Args:
        system_terms (numpy.ndarray): 1D array of functions that define rhs of each ODE in the system. Each function is linearly approximated within a given job.
        approximation_point (numpy.ndarray): 1D array that specifies coordinate around which linear approximation is made.
        sampling_steps (numpy.ndarray): 1D array of steps along each coordinate where additional points are sampled for linear fitting.
        grid (numpy.ndarray): 1D array of x values for which the system terms are evaluated.

    Returns:
        funcs (numpy.ndarray): 3D array with values of approximated rhs terms at all points of this job. 1st dim - equations, 2nd dim - terms, 3rd dim - points.
    """
    funcs = np.zeros((len(system_terms), 1 + len(system_terms), len(grid)))
    if len(grid) == 1:
        # Only shifts need to be calculated
        for eq_ind in range(funcs.shape[0]):
            funcs[eq_ind, 0, 0] = system_terms[eq_ind](grid[0], *approximation_point)
    else:
        fitting_matrix = np.zeros((funcs.shape[1], funcs.shape[1]))
        for row_ind in range(funcs.shape[1]):
            next_point = approximation_point.copy()
            if row_ind > 0:
                next_point[row_ind - 1] += sampling_steps[row_ind - 1]
            fitting_matrix[row_ind, :] = [1, *next_point]

        for eq_ind in range(funcs.shape[0]):
            for point_ind in range(funcs.shape[2]):
                fitting_vector = np.zeros(funcs.shape[1])
                for row_ind in range(funcs.shape[1]):
                    next_point = approximation_point.copy()
                    if row_ind > 0:
                        next_point[row_ind - 1] += sampling_steps[row_ind - 1]
                    fitting_vector[row_ind] = system_terms[eq_ind](grid[point_ind], *next_point)
                funcs[eq_ind, :, point_ind] = np.linalg.solve(fitting_matrix, fitting_vector)
    return funcs


def solve_ode_qp(system_terms, grid, boundary_condition, points_per_step):
    """Solves a given ODE system, defined by system_terms and known_points, by formulating it as a QP problem.

    Args:
        system_terms (numpy.ndarray): 1D array of functions that define rhs of each ODE in the system. Each function is linearly approximated within a given job.
        grid (numpy.ndarray): 1D Array of equidistant grid points.
        boundary_condition (numpy.ndarray): 1D array of initial values for each function in the system.
        points_per_step (int): Number of points to vary per job.

    Returns:
        solution (numpy.ndarray): 2D array with solution for all functions at all points of grid.
        errors (numpy.ndarray): 1D array with errors for each job.
    """
    solution = np.zeros((len(system_terms), len(grid)))
    solution[:, 0] = boundary_condition
    dx = grid[1] - grid[0]
    point_ind = 0
    errors = []
    working_grid = grid[:-1]
    while point_ind < len(working_grid):
        if point_ind == 0:
            sampling_steps = np.zeros(len(system_terms))
            funcs = calculate_term_coefficients(system_terms, solution[:, point_ind], sampling_steps, working_grid[point_ind : point_ind + 1])
        else:
            sampling_steps = solution[:, point_ind] - solution[:, point_ind - 1]
            sampling_steps[abs(sampling_steps) < 1e-10] = 1e-10  # Ensure non-zero steps
            funcs = calculate_term_coefficients(system_terms, solution[:, point_ind], sampling_steps, working_grid[point_ind : point_ind + points_per_step])

        H, d, energy_shift = build_qp_matrices(funcs, dx, solution[:, point_ind])
        solution_points = qpsolvers.solve_qp(H, d)
        errors.append(np.dot(np.matmul(solution_points, H), solution_points) / 2 + np.dot(solution_points, d) + energy_shift)
        solution_points_shaped = np.reshape(solution_points, (len(system_terms), funcs.shape[2]), order='F')
        solution[:, point_ind + 1 : point_ind + funcs.shape[2] + 1] = solution_points_shaped
        point_ind += funcs.shape[2]

    return solution, np.array(errors)


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

    Returns:
        energy_shift (float): Constant part of minimization functional.
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
            unknown_point = scheme_point_global - first_unknown_point_global
            if unknown_point < 0:
                point = bits_to_real(known_bits[scheme_point_global * bits_per_point : (scheme_point_global + 1) * bits_per_point], bits_integer)
                energy_shift += c_factor * point
            else:
                energy_shift -= c_factor * 2 ** (bits_integer - 1)
                for unknown_bit in range(unknown_point * bits_per_point, (unknown_point + 1) * bits_per_point):
                    j = unknown_bit - unknown_point * bits_per_point - bits_integer + 1
                    d[unknown_bit] += c_factor * 2 ** (-j)

    return energy_shift


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

    Returns:
        energy_shift (float): Constant part of minimization functional.
    """
    energy_shift = 0
    bits_per_point = bits_integer + bits_decimal
    first_unknown_point_global = int(len(known_bits) / bits_per_point)
    for deriv_ind1 in range(1, len(funcs_i)):
        deriv_order1, accuracy_order1, scheme_length1, last_scheme_point_global1 = get_deriv_range(deriv_ind1, point_ind, last_unknown_point_global, max_considered_accuracy)
        coeffs1 = get_finite_difference_coefficients(deriv_order1, accuracy_order1)
        for deriv_ind2 in range(1, len(funcs_i)):
            deriv_order2, accuracy_order2, scheme_length2, last_scheme_point_global2 = get_deriv_range(deriv_ind2, point_ind, last_unknown_point_global, max_considered_accuracy)
            coeffs2 = get_finite_difference_coefficients(deriv_order2, accuracy_order2)
            func_factor = funcs_i[deriv_ind1] * funcs_i[deriv_ind2] / dx ** (deriv_order1 + deriv_order2)
            for scheme_point1 in range(scheme_length1):
                scheme_point_global1 = point_ind + scheme_point1
                unknown_point1 = scheme_point_global1 - first_unknown_point_global
                for scheme_point2 in range(scheme_length2):
                    scheme_point_global2 = point_ind + scheme_point2
                    unknown_point2 = scheme_point_global2 - first_unknown_point_global
                    c_factor = func_factor * coeffs1[scheme_point1] * coeffs2[scheme_point2]
                    if unknown_point1 < 0 and unknown_point2 < 0:
                        point1 = bits_to_real(known_bits[scheme_point_global1 * bits_per_point : (scheme_point_global1 + 1) * bits_per_point], bits_integer)
                        point2 = bits_to_real(known_bits[scheme_point_global2 * bits_per_point : (scheme_point_global2 + 1) * bits_per_point], bits_integer)
                        energy_shift += c_factor * point1 * point2
                    else:
                        energy_shift += c_factor * 4 ** (bits_integer - 1)
                        if unknown_point1 >= 0:
                            for unknown_bit in range(unknown_point1 * bits_per_point, (unknown_point1 + 1) * bits_per_point):
                                j = unknown_bit - unknown_point1 * bits_per_point - bits_integer + 1
                                d[unknown_bit] -= c_factor * 2 ** (bits_integer - 1 - j)
                        else:
                            for known_bit in range(scheme_point_global1 * bits_per_point, (scheme_point_global1 + 1) * bits_per_point):
                                j = known_bit - scheme_point_global1 * bits_per_point - bits_integer + 1
                                energy_shift -= c_factor * 2 ** (bits_integer - 1 - j) * known_bits[known_bit]

                        if unknown_point2 >= 0:
                            for unknown_bit in range(unknown_point2 * bits_per_point, (unknown_point2 + 1) * bits_per_point):
                                j = unknown_bit - unknown_point2 * bits_per_point - bits_integer + 1
                                d[unknown_bit] -= c_factor * 2 ** (bits_integer - 1 - j)
                        else:
                            for known_bit in range(scheme_point_global2 * bits_per_point, (scheme_point_global2 + 1) * bits_per_point):
                                j = known_bit - scheme_point_global2 * bits_per_point - bits_integer + 1
                                energy_shift -= c_factor * 2 ** (bits_integer - 1 - j) * known_bits[known_bit]

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

    return energy_shift


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
        energy_shift (float): Constant part of minimization functional.
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
    energy_shift = 0
    for point_ind in range(first_contributing_point, last_contributing_point + 1):
        energy_shift += add_linear_terms_qubo(d, point_ind, last_unknown_point_global, funcs[:, point_ind], dx, known_bits, bits_integer, bits_decimal, max_considered_accuracy)
        energy_shift += add_quadratic_terms_qubo(H, d, point_ind, last_unknown_point_global, funcs[:, point_ind], dx, known_bits, bits_integer, bits_decimal, max_considered_accuracy)
    Q = H + np.diag(d)
    return Q, energy_shift


def solve_ode_qubo(system_terms, grid, known_points, bits_integer, bits_decimal, max_considered_accuracy, points_per_step, sampler, max_attempts, max_error, **kwargs):
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
        max_error (float): Maximum error that does not trigger restart.
        kwargs (dict): Sampler parameters.

    Returns:
        known_points (numpy.ndarray): 2D array with solution for all functions at all points of grid.
        errors (numpy.ndarray): 2D array with errors for each equation (rows) and each solved QUBO (columns).
    """
    known_points = np.vectorize(lambda num: bits_to_real(real_to_bits(num, bits_integer, bits_decimal), bits_integer))(known_points)
    bits_per_point = bits_integer + bits_decimal
    known_bits = np.array([np.concatenate([real_to_bits(elem, bits_integer, bits_decimal) for elem in row]) for row in known_points])
    known_points_extended = np.pad(known_points, [(0, 0), (0, len(grid) - known_points.shape[1])], constant_values=np.nan)
    funcs = np.array([[[term(*args) for args in np.vstack((grid, known_points_extended)).T] for term in equation_terms] for equation_terms in system_terms])
    dx = grid[1] - grid[0]
    errors = [[] for i in range(system_terms.shape[0])]
    with open('log.txt', 'w') as log:
        while known_points.shape[1] < len(grid):
            all_solution_bits_list = []
            all_solution_points_list = []
            for eq_ind in range(system_terms.shape[0]):
                Q, energy_shift = build_qubo_matrix(funcs[eq_ind, :, :], dx, known_bits[eq_ind, :], bits_integer, bits_decimal, max_considered_accuracy, points_per_step)
                lowest_error = np.inf
                solution_bits = np.zeros((1, Q.shape[0]))
                for attempt in range(max_attempts):
                    job_label = f'Point {known_points.shape[1]}; Eq. {eq_ind}; Attempt {attempt + 1}'
                    sample_set = sampler.sample_qubo(Q, label=job_label, **kwargs)
                    samples_plain = np.array([list(sample.values()) for sample in sample_set])  # 2D, each row - solution (all bits together), sorted by energy
                    trial_bits = samples_plain[0, :]
                    solution_error = np.dot(np.matmul(trial_bits, Q), trial_bits) + energy_shift
                    log.write(f'{job_label}; Error {solution_error}\n')

                    if solution_error < lowest_error:
                        lowest_error = solution_error
                        solution_bits = trial_bits

                    if solution_error < max_error:
                        break

                all_solution_bits_list.append(solution_bits)
                solution_bits_shaped = np.reshape(solution_bits, (-1, bits_per_point))
                solution_points = np.array([bits_to_real(row_bits, bits_integer) for row_bits in solution_bits_shaped])
                all_solution_points_list.append(solution_points)
                errors[eq_ind].append(lowest_error)

            all_solution_bits = np.array(all_solution_bits_list)
            all_solution_points = np.array(all_solution_points_list)
            known_bits = np.hstack((known_bits, all_solution_bits))
            known_points = np.hstack((known_points, all_solution_points))
            # Update funcs
            update_cols = range(known_points.shape[1] - all_solution_points.shape[1], known_points.shape[1])
            funcs[:, :, update_cols] = [[[term(*args) for args in np.vstack((grid[update_cols], all_solution_points)).T] for term in equation_terms] for equation_terms in system_terms]

    return known_points, np.array(errors)
