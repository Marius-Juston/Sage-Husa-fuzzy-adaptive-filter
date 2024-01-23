import numpy as np
from line_profiler_pycharm import profile

np.random.seed(42)


@profile
def correct_trapezoid(a):
    c2 = a[:, a.shape[1] - 1]

    for i in range(a.shape[1] - 2, -1, -1):
        c1 = a[:, i]

        mask = c1 < c2
        c1[:] = np.where(mask, c1, c2)

        c2 = c1


@profile
def correct_trapezoid4(a):
    c2 = a[:, a.shape[1] - 1]

    for i in range(a.shape[1] - 2, -1, -1):
        c1 = a[:, i]

        c1[:] = np.where(c1 < c2, c1, c2)

        c2 = c1


@profile
def correct_trapezoid2(a):
    c2 = a[:, a.shape[1] - 1]

    for i in range(a.shape[1] - 2, -1, -1):
        c1 = a[:, i]

        mask = c1 > c2
        c1[mask] = c2[mask]

        c2 = c1


@profile
def correct_trapezoid3(a):
    c2 = a[:, a.shape[1] - 1]

    for i in range(a.shape[1] - 2, -1, -1):
        c1 = a[:, i]

        mask = c1 > c2
        np.copyto(c1, c2, where=mask)

        c2 = c1


@profile
def correct_trapezoid5(a):
    c2 = a[:, a.shape[1] - 1]

    for i in range(a.shape[1] - 2, -1, -1):
        c1 = a[:, i]

        np.copyto(c1, c2, where=c1 > c2)

        c2 = c1


if __name__ == '__main__':
    n = 100000000

    a = np.random.uniform(0, 1000, (n, 4))
    b = a.copy()
    c = a.copy()
    # a = np.random.uniform(0, 1000, (10, 4))

    correct_trapezoid(a)

    # correct_trapezoid2(b)

    # correct_trapezoid3(c)
    correct_trapezoid4(b)

    correct_trapezoid5(c)

    print(np.alltrue(a == c))
