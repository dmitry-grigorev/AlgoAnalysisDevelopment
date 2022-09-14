import numpy as np

phi = (1 + np.sqrt(5)) / 2
invphi = (-1 + np.sqrt(5)) / 2
invphi2 = (3 - np.sqrt(5)) / 2


def exhaustive_search(func, left=0, right=1, eps=1e-3):
    niters, fcalcs = 0, 0
    grid = np.linspace(left, right, int((right - left) / eps) + 1).tolist()
    xmin = left
    fmin = func(left)
    fcalcs += 1
    for x in grid[1:]:
        niters += 1
        f = func(x)
        fcalcs += 1
        if f < fmin:
            xmin, fmin = x, f
    return xmin, fmin, niters, fcalcs


def dichotomy_method(func, left=0, right=1, eps=1e-3):
    niters, fcalcs = 0, 0
    while right - left > 2 * eps:
        niters += 1
        m = (left + right) / 2
        a, b = m - eps / 2, m + eps / 2
        if func(a) < func(b):
            right = b
        else:
            left = a
        fcalcs += 2
    fcalcs += 1
    xmin = (left + right) / 2
    return xmin, func(xmin), niters, fcalcs


def goldenratio_method(func, left=0, right=1, eps=1e-3):
    niters, fcalcs = 0, 0
    h = right - left
    a = right - h * invphi
    b = left + h * invphi
    fa = func(a)
    fb = func(b)
    fcalcs += 2
    while h > eps:
        niters += 1
        h = invphi * h
        if fa < fb:
            right = b
            b = a
            fb = fa
            a = right - h * invphi
            fa = func(a)
        else:
            left = a
            a = b
            fa = fb
            b = left + h * invphi
            fb = func(b)
        fcalcs += 1
        h = right - left
    xmin = (a + b) / 2
    fcalcs += 1
    return xmin, func(xmin), niters, fcalcs


def exhaustive_search2d(func, leftx=0, rightx=1, lefty=0, righty=1, eps=1e-3, optpoint=None):
    gridx = np.linspace(leftx, rightx, int((rightx - leftx) / eps) + 1).tolist()
    gridy = np.linspace(lefty, righty, int((righty - lefty) / eps) + 1).tolist()

    niters, fcalcs = 0, 0
    xmin, ymin = gridx[0], gridy[0]
    fmin = np.infty

    for i in range(len(gridx)):
        for j in range(len(gridy)):
            niters += 1
            f = func(gridx[i], gridy[j])
            fcalcs += 1
            if f < fmin:
                xmin, ymin, fmin = gridx[i], gridy[j], f

    if optpoint is None:
        return (xmin, ymin), fmin, niters, fcalcs, None
    else:
        return (xmin, ymin), fmin, niters, fcalcs, np.sqrt((xmin - optpoint[0])**2 + (ymin - optpoint[1])**2)


def coordinate_descent_method(func, alpha=0, beta=1, max_iterations=1000, eps=1e-3, optpoint=None):
    arg_cur = (0, 0)
    arg_prev = (0, 0)
    func_res_cur = 0
    func_res_prev = 0
    i = 0
    func_calculations = 0
    iterations = 0
    # there is case when coordinate descent can run forever, so there is additional condition
    # to eliminate this problem, param max_iterations defines how many iterations
    # will be executed if stop conditions are not met
    while max_iterations > i:
        i += 1
        temp_arg = arg_cur
        res = None
        # switching to another variable as const on every iteration
        if i % 2 == 0:
            # using x2 as const
            res = dichotomy_method(lambda x: func(x, arg_cur[1]), alpha, beta, eps)
            # res[0] = min x
            arg_cur = (res[0], arg_cur[1])
        else:
            # using x1 as const
            res = dichotomy_method(lambda x: func(arg_cur[0], x), alpha, beta, eps)
            # res[0] = min x
            arg_cur = (arg_cur[0], res[0])

        func_res_prev = func_res_cur
        func_res_cur = res[1]
        # including current while loop iterations
        iterations += res[2] + 1
        func_calculations += res[3]

        arg_prev = temp_arg

        # stop condition - (x[k + 1] - x[k] <= eps) OR (f(x[k+1]) - f(x[k]) <= eps)
        if coord_diff_less_equal_to_eps(arg_prev, arg_cur, eps) or \
                abs(func_res_cur - func_res_prev) <= eps:
            break

    if i == max_iterations:
        print("Solution probably not found, break out of loop due to reaching max allowed iterations ("
              + str(max_iterations) + ")")

    if optpoint is None:
        return arg_cur, func(arg_cur[0], arg_cur[1]), iterations, func_calculations, None
    else:
        return arg_cur, func(arg_cur[0], arg_cur[1]), iterations, func_calculations, \
               np.sqrt((arg_cur[0] - optpoint[0])**2 + (arg_cur[1] - optpoint[1])**2)


# returns true if difference between each coordinate is less or equal to given eps
def coord_diff_less_equal_to_eps(left, right, eps):
    return abs(right[0] - left[0]) <= eps and abs(right[1] - left[1]) <= eps


def squares(x: np.ndarray, y: np.ndarray):
    return np.sum(np.square(x - y))
