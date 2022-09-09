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

    while low < high:
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

    v[start], v[high] = v[high], v[start]
    return high


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
