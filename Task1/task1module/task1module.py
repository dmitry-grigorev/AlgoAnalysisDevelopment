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


def quick_sort(v):
    quick_sort_r(v, 0, len(v) - 1)


def quick_sort_r(v, start, end):
    if start >= end:
        return

    p = partition(v, start, end)
    quick_sort_r(v, start, p - 1)
    quick_sort_r(v, p + 1, end)


def partition(v, start, end):
    pivot = v[start]
    low = start + 1
    high = end

    while True:
        # If the current value we're looking at is larger than the pivot
        # it's in the right place (right side of pivot) and we can move left,
        # to the next element.
        # We also need to make sure we haven't surpassed the low pointer, since that
        # indicates we have already moved all the elements to their correct side of the pivot
        while low <= high and v[high] >= pivot:
            high = high - 1

        # Opposite process of the one above
        while low <= high and v[low] <= pivot:
            low = low + 1

        # We either found a value for both high and low that is out of order
        # or low is higher than high, in which case we exit the loop
        if low < high:
            v[low], v[high] = v[high], v[low]
        else:
            break

    v[start], v[high] = v[high], v[start]
    return high


def tim_sort(v):
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
