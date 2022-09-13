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
