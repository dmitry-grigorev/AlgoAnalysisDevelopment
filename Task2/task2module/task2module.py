import numpy as np

phi = (1 + np.sqrt(5)) / 2
invphi = (-1 + np.sqrt(5)) / 2
invphi2 = (3 - np.sqrt(5)) / 2


def exhaustive_search(func, left=0, right=1, eps=1e-3):
    niters, fcalcs = 0, 0
    grid = np.linspace(left, right, int(1 / eps) + 1)
    xmin = left
    fmin = func(left)
    fcalcs += 1
    for x in grid[1:]:
        niters += 1
        f = func(x)
        fcalcs += 1
        if f < fmin:
            xmin = x
            fmin = f
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


def exhaustive_search2d(func, leftx=0, rightx=1, lefty=0, righty=1, eps=1e-3):
    gridx = np.linspace(leftx, rightx, int(1 / eps) + 1)
    gridy = np.linspace(lefty, righty, int(1 / eps) + 1)

    # to do: make gridx x gridy product, calculate func on it and finc minimum and its point

    gridxy = np.meshgrid(gridx, gridy)
    gridf = func(*gridxy)

    jmin, imin = np.where(gridf == np.min(gridf))

    xmin, ymin = gridx[imin], gridy[jmin]

    return xmin[0], ymin[0], func(xmin, ymin)[0]


def coordinate_descent_method(func, alpha, beta, max_iterations, eps=1e-3):
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
        i = i + 1
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
        if coord_diff_less_equal_to_eps(arg_prev, arg_cur, eps) or\
                abs(func_res_cur - func_res_prev) <= eps:
            break

    if i == max_iterations:
        print("Solution probably not found, break out of loop due to reaching max allowed iterations ("
              + str(max_iterations) + ")")

    return arg_cur, func(arg_cur[0], arg_cur[1]), iterations, func_calculations


# returns true if difference between each coordinate is less or equal to given eps
def coord_diff_less_equal_to_eps(left, right, eps):
    return abs(right[0] - left[0]) <= eps and abs(right[1] - left[1]) <= eps
