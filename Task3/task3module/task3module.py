import numpy as np


def squares(x: np.ndarray, y: np.ndarray):
    return np.sum(np.square(x - y))


def belongs(x: np.ndarray, area: np.ndarray):
    return np.all(x > area[:, 0]) and np.all(x < area[:, 1])


def newtons(func, limits, x0, eps=1e-3, funcgrad=None, funchess=None, optpoint=None):
    # if funcgrad is None:
    #    funcgrad = autograd.elementwise_grad(func)
    # if funchess is None:
    #    funchess = autograd.elementwise_grad(funcgrad)
    next, prev = x0, None
    niters = 0
    while (prev is None or np.linalg.norm(next - prev) > eps) and belongs(next, limits):
        g, H = funcgrad(next), funchess(next)
        delta = -np.linalg.inv(H) @ g
        prev = next
        next = next + delta
        niters += 1

    if optpoint is None:
        return next, func(next), niters, None, None
    else:
        return next, func(next), niters, None, np.linalg.norm(next - optpoint)
