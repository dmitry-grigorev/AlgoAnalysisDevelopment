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


def calc_polynom_horner(coefs: np.ndarray, x):
    s = coefs[-1]
    for coef in coefs[::-1][1:]:
        s = coef + x * s
    return s


def bubble_sort(v: np.ndarray):
    has_swapped = True
    num_of_iterations = 0

    while has_swapped:
        has_swapped = False
        for i in range(v.shape[0] - num_of_iterations - 1):
            if v[i] > v[i + 1]:
                v[i], v[i + 1] = v[i + 1], v[i]
                has_swapped = True
        num_of_iterations += 1
    return v


def quick_sort(v: np.ndarray):
    quick_sort_r(v, 0, v.shape[0] - 1)


def quick_sort_r(v, start, end):
    if len(v) == 1:
        return
    if start < end:
        pi = partition(v, start, end)
        quick_sort_r(v, start, pi - 1)  # Recursively sorting the left values
        quick_sort_r(v, pi + 1, end)  # Recursively sorting the right values


def partition(v, start, end):
    middle = (start + end - 1) // 2
    median_first(v, start, middle, end - 1)
    pivot = v[start]
    i = start + 1
    for j in range(start + 1, end, 1):
        if v[j] < pivot:
            v[i], v[j] = v[j], v[i]
            i += 1
    v[start], v[i - 1] = v[i - 1], v[start]
    return i - 1


def compare_swap(L, a, b):
    L[a], L[b] = min(L[a], L[b]), max(L[a], L[b])


def median_first(L, a, b, c):
    compare_swap(L, b, a)  # L[b]<=L[a]
    compare_swap(L, b, c)  # L[b]<=L[c] i.e L[b] smallest
    compare_swap(L, a, c)  # L[a]<=L[c] i.e L[c] largest


def calc_min_run(length, minMerge):
    """Returns the minimum length of a
        run from 23 to 64 so that
        the len(array)/minrun is less than or
        equal to a power of 2.

        e.g. 1=>1, ..., 63=>63, 64=>32, 65=>33,
        ..., 127=>64, 128=>32, ...
        """
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
    # original array is broken in two parts
    # left and right array
    len1 = mid - left + 1
    len2 = right - mid
    leftArr, rightArr = [], []
    for i in range(0, len1):
        leftArr.append(v[left + i])
    for i in range(0, len2):
        rightArr.append(v[mid + 1 + i])

    i, j, k = 0, 0, left

    # after comparing, we merge those two array
    # in larger sub array
    while i < len1 and j < len2:
        if leftArr[i] <= rightArr[j]:
            v[k] = leftArr[i]
            i += 1

        else:
            v[k] = rightArr[j]
            j += 1

        k += 1

    # Copy remaining elements of left, if any
    while i < len1:
        v[k] = leftArr[i]
        k += 1
        i += 1

    # Copy remaining element of right, if any
    while j < len2:
        v[k] = rightArr[j]
        k += 1
        j += 1


def tim_sort(v):
    length = len(v)
    minRun = calc_min_run(length, 32)
    for start in range(0, length, minRun):
        end = min(start + minRun - 1, length - 1)
        insertion_sort(v, start, end)

    size = minRun
    while size < length:

        # Pick starting point of left sub array. We
        # are going to merge arr[left..left+size-1]
        # and arr[left+size, left+2*size-1]
        # After every merge, we increase left by 2*size
        for left in range(0, length, 2 * size):

            # Find ending point of left sub array
            # mid+1 is starting point of right sub array
            mid = min(length - 1, left + size - 1)
            right = min((left + 2 * size - 1), (length - 1))

            # Merge sub array arr[left.....mid] &
            # arr[mid+1....right]
            if mid < right:
                merge(v, left, mid, right)

        size = 2 * size


def matrix_multiplication(A: np.ndarray, B: np.ndarray):
    n = A.shape[0]
    C = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


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
    print(fit[0])
    return ggplot() + geom_point(aes(x=n_seq, y=times)) + xlab("Array size") + \
           ylab("Execution time (in milliseconds)") + geom_line(aes(x=n_seq, y=fit[1] + fit[0] * fitfunc(n_seq)),
                                                                color="red", size=2, alpha=0.5)


def plot_algo_complexity_info(func, fitfunc, args=list()):
    times = np.zeros(2000, dtype=float)
    for _ in range(5):
        times += test_time_complexity(func, [np.random.uniform(size=2000)] + args, 2000)
    times /= 5
    return plot_exectime(times, fitfunc)


def test_mul_complexity(A: np.ndarray, B: np.ndarray):
    n = A.shape[0]
    exectime = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        start = timeit.default_timer()
        matrix_multiplication(A[:i, :i], B[:i, :i])
        exectime[i - 1] = (timeit.default_timer() - start) * 1000
    return exectime


def plot_mul_complexity_info():
    times = np.zeros(100, dtype=float)
    for _ in range(5):
        times += test_mul_complexity(np.random.uniform(size=(100, 100)), np.random.uniform(size=(100, 100)))
    times /= 5
    return plot_exectime(times, lambda x: x ** 3)
