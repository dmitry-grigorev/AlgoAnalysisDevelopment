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
    s, k = 0, 0
    for coef in coefs:
        s += coef * np.power(x, k)
        k += 1
    return s


def fake_cacl_polynom_direct(coefs, x):
    s, k = 0, 0
    for coef in coefs:
        i = 0
        while i < k:
            i += 1
            x * x
        s += coef * x
        k += 1
    return s


def calc_polynom_horner(coefs: np.ndarray, x):
    s = coefs[-1]
    for coef in coefs[::-1][1:]:
        s = coef + x * s
    return s


def bubble_sort(v):
    has_swapped = True
    num_of_iterations = 0

    while has_swapped:
        has_swapped = False
        for i in range(len(v) - num_of_iterations - 1):
            if v[i] > v[i + 1]:
                v[i], v[i + 1] = v[i + 1], v[i]
                has_swapped = True
        num_of_iterations += 1
    return v


def quick_sort(v):
    quick_sort_r(v, 0, len(v) - 1)


def quick_sort_r(v, start, end):
    if len(v) == 1:
        return

    size = end - start + 1
    stack = [0] * size

    top = -1

    top = top + 1
    stack[top] = start
    top = top + 1
    stack[top] = end

    while top >= 0:

        # Pop h and l
        end = stack[top]
        top = top - 1
        start = stack[top]
        top = top - 1

        p = partition(v, start, end)

        if p - 1 > start:
            top = top + 1
            stack[top] = start
            top = top + 1
            stack[top] = p - 1

        if p + 1 < end:
            top = top + 1
            stack[top] = p + 1
            top = top + 1
            stack[top] = end


def partition(v, start, end):
    i = (start - 1)
    x = v[end]

    for j in range(start, end):
        if v[j] <= x:
            i = i + 1
            v[i], v[j] = v[j], v[i]

    v[i + 1], v[end] = v[end], v[i + 1]
    return (i + 1)


def calc_min_run(length, minMerge):
    r = 0
    while length >= minMerge:
        r |= length & 1
        length >>= 1
    return length + r


def insertion_sort(v, left, right):
    for i in range(left + 1, right + 1):
        j = i
        while j > left and v[j] < v[j - 1]:
            v[j], v[j - 1] = v[j - 1], v[j]
            j -= 1


def merge(v, left, mid, right):
    len1 = mid - left + 1
    len2 = right - mid
    leftArr, rightArr = [0] * len1, [0] * len2
    for i in range(0, len1):
        leftArr[i] = v[left + i]
    for i in range(0, len2):
        rightArr[i] = v[mid + 1 + i]

    i, j, k = 0, 0, left

    while i < len1 and j < len2:
        if leftArr[i] <= rightArr[j]:
            v[k] = leftArr[i]
            i += 1

        else:
            v[k] = rightArr[j]
            j += 1

        k += 1

    while i < len1:
        v[k] = leftArr[i]
        k += 1
        i += 1

    while j < len2:
        v[k] = rightArr[j]
        k += 1
        j += 1


def tim_sort(v: np.ndarray):
    length = len(v)
    minRun = calc_min_run(length, 32)
    for start in range(0, length, minRun):
        end = min(start + minRun - 1, length - 1)
        insertion_sort(v, start, end)

    size = minRun
    while size < length:

        for left in range(0, length, 2 * size):

            mid = min(length - 1, left + size - 1)
            right = min((left + 2 * size - 1), (length - 1))

            if mid < right:
                merge(v, left, mid, right)

        size = 2 * size


def matrix_multiplication(A, B):
    n = len(A)
    C = [[0]*n]*n
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def test_time_complexity(function, args, n):
    exectime = np.zeros(n, dtype=float)
    v, fargs = args[0].tolist(), args[1:]
    for i in range(1, n + 1):
        vect = v[:i]
        start = timeit.default_timer()
        function(vect, *fargs)
        exectime[i - 1] = (timeit.default_timer() - start) * 1000
    return exectime


def plot_exectime(times: np.ndarray, fitfunc):
    n = times.shape[0]
    n_seq = np.linspace(1, n, n)
    fit = np.polyfit(fitfunc(n_seq), times, 1)
    print(fit[0])
    return fit[0], ggplot() + geom_point(aes(x=n_seq, y=times)) + xlab("Array size") + \
           ylab("Execution time (in milliseconds)") + geom_line(aes(x=n_seq, y=fit[1] + fit[0] * fitfunc(n_seq)),
                                                                color="red", size=2, alpha=0.5)


def plot_algo_complexity_info(func, fitfunc, args=list()):
    times = np.zeros(2000, dtype=float)
    for _ in range(5):
        times += test_time_complexity(func, [np.random.uniform(size=2000)] + args, 2000)
    return plot_exectime(times / 5, fitfunc)


def test_mul_complexity(A, B):

    n = len(A)
    listA, listB = A.tolist(), B.tolist()
    exectime = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        start = timeit.default_timer()
        matrix_multiplication(listA[:i][:i], listB[:i][:i])
        exectime[i - 1] = (timeit.default_timer() - start) * 1000
    return exectime


def plot_mul_complexity_info():
    times = np.zeros(100, dtype=float)
    for _ in range(5):
        times += test_mul_complexity(np.random.uniform(size=(100, 100)), np.random.uniform(size=(100, 100)))
    return plot_exectime(times / 5, lambda x: x ** 3)
