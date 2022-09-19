import numpy as np
from Task2.task2module.task2module import goldenratio_method
from _collections_abc import Sequence


def squares(x: np.ndarray, y: np.ndarray):
    return np.sum(np.square(x - y))


def belongs(x: np.ndarray, area: np.ndarray):
    return np.all(x > area[:, 0]) and np.all(x < area[:, 1])


"""
This function implements Newton's method of optimization.
Function 'func' have to be vectorized over its arguments.
limits describes rectangle, it is an array with n x 2 dimensions.
If optimal point 'optpoint' is provided, this function provides l2-norm precision. 
"""


def newtons(func, limits, x0, funcgrad, funchess, eps=1e-3, optpoint=None):
    next, prev = x0, None
    niters, fcalcs, gradcalcs, hesscalcs, matrixinvers = 0, 0, 0, 0, 0
    while (prev is None or np.linalg.norm(next - prev) > eps) and belongs(next, limits):
        g, H = funcgrad(next), funchess(next)
        delta = -np.linalg.inv(H) @ g
        prev = next
        next = next + delta
        niters += 1
        matrixinvers += 1
        gradcalcs += 1
        hesscalcs += 1

    calcsstats = {"iterations": niters,
                  "funccalcs": fcalcs,
                  "gradcalcs": gradcalcs,
                  "hesscalcs": hesscalcs,
                  "matrixinversions": matrixinvers}

    if optpoint is None:
        return next, func(next), calcsstats, None
    else:
        return next, func(next), calcsstats, np.linalg.norm(next - optpoint)


"""
This function implements Levenberg-Marquardt method of optimization.
Function 'func' have to be vectorized over its arguments.
limits describes rectangle, it is an array with n x 2 dimensions.
Parameter of this method regulpar might be either a number or a tuple two numbers.
In the case of tuple, this parameter specify the segment for line-search of optimal parameter.
If optimal point 'optpoint' is provided, this function provides l2-norm precision. 
"""


def levenmarq(func, limits, x0, regulpar: Sequence[int, float, tuple[float, float]],
              funcgrad, funchess, eps=1e-3, optpoint=None):
    next, prev = x0, None
    delta = None
    niters, fcalcs, gradcalcs, hesscalcs, matrixinvers = 0, 0, 0, 0, 0
    n = x0.shape[0]
    identitymatrix = np.eye(N=n, dtype=float)

    while (prev is None or np.linalg.norm(next - prev) > eps) and belongs(next, limits):
        g, H = funcgrad(next), funchess(next)
        gradcalcs += 1
        hesscalcs += 1
        if isinstance(regulpar, float) or isinstance(regulpar, int):
            delta = -np.linalg.inv(H + regulpar * identitymatrix) @ g
            matrixinvers += 1
        elif isinstance(regulpar, tuple):
            linesearchres = goldenratio_method(lambda nu: func(-np.linalg.inv(H + nu * identitymatrix) @ g),
                                               left=regulpar[0], right=regulpar[1])
            fcalcs += linesearchres[3]
            delta = -np.linalg.inv(H + linesearchres[0] * identitymatrix) @ g
            matrixinvers += (1 + linesearchres[2])

        prev = next
        next = next + delta
        niters += 1

    calcsstats = {"iterations": niters,
                  "funccalcs": fcalcs,
                  "gradcalcs": gradcalcs,
                  "hesscalcs": hesscalcs,
                  "matrixinversions": matrixinvers}

    if optpoint is None:
        return next, func(next), calcsstats, None
    else:
        return next, func(next), calcsstats, np.linalg.norm(next - optpoint)
