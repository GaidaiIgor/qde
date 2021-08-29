"""This module contains functions that solve differential equations by transforming them to QUBO problems, which allows solution on quantum annealer."""
from dwave.system.composites import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwave_qbsolv import QBSolv
import numpy as np
import qpsolvers


class QUBOSampler:
    """Base class for implementations of QUBO sampling approaches."""

    def sample_qubo(self, Q, label=''):
        """Returns a sample set of found binary vectors that minimizes QUBO functional.

        Args:
            Q (numpy.ndarray): QUBO minimization matrix.
            label (str): Optional sampling job label.

        Returns:
            sample_set (dimod.SampleSet): Sample set of found binary vectors that minimizes QUBO functional.
        """
        raise NotImplementedError("Not implemented in base class")


class QBSolvWrapper(QUBOSampler):
    """Uses classical QBSolv sampling algorithm (probabilistic heuristic)."""

    def __init__(self, num_repeats):
        """Initializes instance.

        Args:
            num_repeats (int): Number of times to repeat sampling procedure to attempt to find a better solution.
        """
        self.num_repeats = num_repeats

    def sample_qubo(self, Q, label=''):
        """See base class."""
        sample_set = QBSolv().sample_qubo(Q, label=label, num_repeats=self.num_repeats)
        return sample_set


class DWaveSamplerWrapper(QUBOSampler):
    """Uses D-Wave quantum annealer for sampling."""

    def __init__(self, num_reads):
        """Initializes instance.

        Args:
            num_reads (int): Number of times ground state of qubits is read.
        """
        self.num_reads = num_reads

    def sample_qubo(self, Q, label=''):
        """See base class."""
        sample_set = EmbeddingComposite(DWaveSampler()).sample_qubo(Q, label=label, num_reads=self.num_reads)
        return sample_set


class Solver:
    """Base class for quadratic minimization solvers."""

    def solve(self, H, d, job_label=''):
        """Finds solution of quadratic minimization problem.

        Args:
            H (numpy.ndarray): Quadratic minimization matrix.
            d (numpy.ndarray): Quadratic minimization vector.
            job_label (str): Optional job label.

        Returns:
            solution (numpy.ndarray): 1D array that minimizes QP functional.
        """
        raise NotImplementedError("Not implemented in base class")


class QPSolver(Solver):
    """Solves quadratic minimization problem with real variables."""

    def solve(self, H, d, job_label=''):
        """See base class."""
        solution = qpsolvers.solve_qp(2*H, d)
        return solution


class QUBOSolver(Solver):
    """Solves quadratic minimization problem with binary variables."""

    def __init__(self, bits_integer, bits_decimal, sampler):
        """Initializes instance.

        Args:
            bits_integer (int): Number of binary variables used to represent integer part of a given real number.
            bits_decimal (int): Number of binary variables used to represent decimal part of a given real number.
            sampler (QUBOSampler): Sampler to use for QUBO sampling.
        """
        self.bits_integer = bits_integer
        self.bits_decimal = bits_decimal
        self.bits_total = bits_integer + bits_decimal
        self.sampler = sampler

    def get_discretization_matrix(self):
        """Builds the discretization matrix for given number of bits in integer and decimal parts.

        Returns:
            numpy.ndarray: Discretization matrix.
        """
        j_range = range(-self.bits_integer + 1, self.bits_decimal + 1)
        return np.reshape([2 ** -(j1 + j2) for j1 in j_range for j2 in j_range], (len(j_range), len(j_range)))

    def get_discretization_vector(self):
        """Builds the discretization vector for given number of bits in integer and decimal parts.

        Returns:
            numpy.ndarray: Discretization vector.
        """
        j_range = range(-self.bits_integer + 1, self.bits_decimal + 1)
        return np.array([2 ** -j for j in j_range])

    def real_to_bits(self, num):
        """Returns the closest binary representation of a given real number.

        Args:
            num (float): Number to convert.

        Returns:
            bits (numpy.ndarray): 1D array of bits.
        """
        bits = np.zeros(self.bits_total, dtype=int)
        represented = -2 ** (self.bits_integer - 1)
        for i in range(len(bits)):
            bit_value = 2 ** (self.bits_integer - 1 - i)
            if represented + bit_value <= num:
                bits[i] = 1
                represented += bit_value
        return bits

    def bits_to_real(self, bits):
        """Returns a real number represented by given binary representation.

        Args:
            bits (numpy.ndarray): 1D array of bits.

        Returns:
            num: Represented real number.
        """
        discretization_vector = self.get_discretization_vector()
        return np.dot(bits, discretization_vector) - 2 ** (self.bits_integer - 1)

    def convert_qp_matrices_to_qubo(self, H, d):
        """Converts QP matrices to QUBO representation with given number of bits for integer and decimal parts.

        Args:
            H (numpy.ndarray): Quadratic minimization matrix.
            d (numpy.ndarray): Quadratic minimization vector.

        Returns:
            Q (numpy.ndarray): Equivalent QUBO matrix.
            energy_shift (float): Additional energy shift for QUBO formulation.
        """
        discretization_matrix = self.get_discretization_matrix()
        discretization_vector = self.get_discretization_vector()
        block_size = len(discretization_vector)
        Q_size = block_size * len(d)
        Q = np.zeros((Q_size, Q_size))
        for i in range(H.shape[0]):
            for j in range(i, H.shape[1]):
                coeff = H[i, j] if i == j else 2 * H[i, j]
                Q[i * block_size: (i + 1) * block_size, j * block_size: (j + 1) * block_size] += coeff * discretization_matrix
                Q[range(i * block_size, (i + 1) * block_size), range(i * block_size, (i + 1) * block_size)] -= 2 ** (self.bits_integer - 1) * coeff * discretization_vector
                Q[range(j * block_size, (j + 1) * block_size), range(j * block_size, (j + 1) * block_size)] -= 2 ** (self.bits_integer - 1) * coeff * discretization_vector
                if i == j:
                    Q[range(i * block_size, (i + 1) * block_size), range(i * block_size, (i + 1) * block_size)] += d[i] * discretization_vector

        energy_shift = 4 ** (self.bits_integer - 1) * np.sum(H) - 2 ** (self.bits_integer - 1) * np.sum(d)
        return Q, energy_shift

    def solve(self, H, d, job_label=''):
        """See base class."""
        Q = self.convert_qp_matrices_to_qubo(H, d)[0]
        sample_set = self.sampler.sample_qubo(Q, label=job_label)
        samples_plain = np.array([list(sample.values()) for sample in sample_set])  # 2D, each row - solution_real (all bits together), sorted by energy
        solution_bits = samples_plain[0, :]
        solution_bits_shaped = np.reshape(solution_bits, (H.shape[0], self.bits_total))
        solution_real = np.array([self.bits_to_real(bits_row) for bits_row in solution_bits_shaped])
        return solution_real


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


def add_point_terms_qp(H, d, point_ind, eq_ind_start, eq_ind_end, funcs_i, dx, known_points=None):
    """Adds functional terms for a given point to H and d.

    Args:
        H (numpy.ndarray): Current quadratic minimization matrix to which quadratic terms of specified point are added.
        d (numpy.ndarray): Current quadratic minimization vector to which linear terms of specified point are added.
        point_ind (int): Local point index within the current job.
        eq_ind_start (int): Index of the first considered equation.
        eq_ind_end (int): Index of the last considered equation (exclusive).
        funcs_i (numpy.ndarray): 2D array with values of approximated rhs terms at the current point. Equations are along rows, terms along columns.
        dx (float): Grid step.
        known_points (numpy.ndarray): When adding terms for the last known point, this is 1D array of the values of each function at that point, otherwise not needed.

    Returns:
        energy_shift (float): Constant part of minimization functional.
    """
    energy_shift = 0
    get_unknown_ind = lambda point, eq: (point - 1) * (eq_ind_end - eq_ind_start) + (eq - eq_ind_start)
    for eq_ind in range(eq_ind_start, eq_ind_end):
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


def build_qp_matrices(funcs, dx, known_points, eq_ind_start, eq_ind_end):
    """Builds matrices H and d that define quadratic minimization problem corresponding to a given system of differential equations.

    Args:
        funcs (numpy.ndarray): 3D array with values of approximated rhs terms at all points of this job. 1st dim - equations, 2nd dim - terms, 3rd dim - points.
        dx (float): Grid step.
        known_points (numpy.ndarray): 1D array of known points for each function at 0th point (boundary condition).
        eq_ind_start (int): Index of the first considered equation.
        eq_ind_end (int): Index of the last considered equation (exclusive).

    Returns:
        H (numpy.ndarray): Quadratic minimization matrix.
        d (numpy.ndarray): Quadratic minimization vector.
        energy_shift (float): Constant part of minimization functional.
    """
    unknowns = (eq_ind_end - eq_ind_start) * funcs.shape[2]
    H = np.zeros((unknowns, unknowns))
    d = np.zeros(unknowns)
    energy_shift = 0
    for point_ind in range(funcs.shape[2]):
        energy_shift += add_point_terms_qp(H, d, point_ind, eq_ind_start, eq_ind_end, funcs[:, :, point_ind], dx, known_points)
    return H, d, energy_shift


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


def solve_ode(system_terms, grid, boundary_condition, points_per_step, equations_per_step, solver, max_attempts, max_error):
    """Solves a given ODE system, defined by system_terms and known_points, by formulating it as a QP problem.

    Args:
        system_terms (numpy.ndarray): 1D array of functions that define rhs of each ODE in the system. Each function is linearly approximated within a given job.
        grid (numpy.ndarray): 1D Array of equidistant grid points.
        boundary_condition (numpy.ndarray): 1D array of initial values for each function in the system.
        points_per_step (int): Number of points to vary per job.
        equations_per_step (int): Number of equations to vary per job.
        solver (Solver): Solver to solve QP problem.
        max_attempts (int): Maximum number of times each problem can be solved (restarts can find a better solution for some solvers).
        max_error (float): Maximum error that does not trigger restart.

    Returns:
        solution (numpy.ndarray): 2D array with solution for all functions at all grid points.
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

        solution_points = np.zeros(solution.shape[0])
        eq_ind = 0
        while eq_ind < len(solution_points):
            H, d, energy_shift = build_qp_matrices(funcs, dx, solution[:, point_ind], eq_ind, eq_ind + equations_per_step)
            lowest_error = np.inf
            for attempt in range(max_attempts):
                job_label = f'Point ind: {point_ind}; Eq. {eq_ind}; Attempt {attempt + 1}'
                trial_points = solver.solve(H, d, job_label)
                trial_error = np.dot(np.matmul(trial_points, H), trial_points) + np.dot(trial_points, d) + energy_shift
                if trial_error < lowest_error:
                    lowest_error = trial_error
                    solution_points[eq_ind : eq_ind + equations_per_step] = trial_points
                if trial_error < max_error:
                    break
            errors.append(lowest_error)
            eq_ind += equations_per_step

        solution_points_shaped = np.reshape(solution_points, (len(system_terms), funcs.shape[2]), order='F')
        solution[:, point_ind + 1 : point_ind + funcs.shape[2] + 1] = solution_points_shaped
        point_ind += funcs.shape[2]

    return solution, np.array(errors)
