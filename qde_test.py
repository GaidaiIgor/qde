import numpy as np
from qde import solve

def test1():
    """Solves dy/dx = exp(x) by formulating it as a QUBO problem."""
    grid_from = 0
    grid_to = 0.1
    N = 2
    y1 = 1
    qbits_integer = 1
    qbits_decimal = 10

    grid = np.linspace(grid_from, grid_to, N)
    f = np.exp(grid)
    dx = grid[1] - grid[0]
    ans = solve(f, dx, y1, qbits_integer, qbits_decimal)
    print(ans)


def test2():
    Q = {('q1', 'q1'): 0.1, ('q2', 'q2'): 0.1, ('q1', 'q2'): -0.2}
    res = QBSolv().sample_qubo(Q)
    print(res)


if __name__ == '__main__':
    np.set_printoptions(linewidth = 200)
    test1()
