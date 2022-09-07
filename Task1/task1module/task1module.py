import numpy as np
import timeit
from plotnine import *


def xlogx(x):
    return x * np.log(x)


def constant(v):
    return 42


def sum_of_elements(v):
    s = 0
    for e in v:
        s += e
    return s


def prod_of_elements(v):
    p = 1
    for e in v:
        p *= e
    return p


def cacl_polynom_direct(coefs, x):
    s, x_ = 0, 1
    for coef in coefs:
        s += coef * x_
        x_ *= x
    return s


def calc_polynom_horner(coefs: np.ndarray, x):
    s = coefs[-1]
    for coef in coefs[::-1][1:]:
        s = coef + x * s
    return s


def bubble_sort(v: np.ndarray):
    has_swapped = True
    num_of_iterations = 0

    while (has_swapped):
        has_swapped = False
        for i in range(v.shape[0] - num_of_iterations - 1):
            if v[i] > v[i + 1]:
                v[i], v[i + 1] = v[i + 1], v[i]
                has_swapped = True
        num_of_iterations += 1
    return v


def test_time_complexity(function, args, n):
    exectime = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        start = timeit.default_timer()
        function((args[0])[:i], *(args[1:]))
        exectime[i - 1] = (timeit.default_timer() - start) * 1000
    return exectime


def plot_exectime(times: np.ndarray, fitfunc):
    n = times.shape[0]
    n_seq = np.linspace(1, n, n)
    fit = np.polyfit(fitfunc(n_seq), times, 1)
    return ggplot() + geom_point(aes(x = n_seq, y = times)) + theme_light() + xlab("Array size") + \
           ylab("Execution time (in milliseconds)") + geom_line(aes(x = n_seq, y = fit[1] + fit[0] * fitfunc(n_seq)),
                                                                color = "red", size = 2)


def plot_algo_complexity_info(func, fitfunc, args = list()):
    times = np.zeros(2000, dtype=float)
    for _ in range(5):
        times += test_time_complexity(func, [np.random.uniform(size=2000)] + args, 2000)
    times /= 5
    return plot_exectime(times, fitfunc)
