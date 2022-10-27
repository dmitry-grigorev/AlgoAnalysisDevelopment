import numpy as np
from Task2.task2module.task2module import goldenratio_method
from _collections_abc import Sequence


def squares(x, y):
    return np.sum(np.square(x - y))


def belongs(x: np.ndarray, area: np.ndarray):
    return np.all(x > area[:, 0]) and np.all(x < area[:, 1])


def projectiononborder(x: np.ndarray, area):
    areacenter = (area[:, 0] + area[:, 1]) / 2
    y = np.subtract(x, areacenter)
    absx = np.abs(y)
    imaxdiff = np.where(absx == np.max(absx))
    maxdiff = absx[imaxdiff]
    size = (area[:, 1] - area[:, 0])[imaxdiff]
    return areacenter + y * size / maxdiff / 2


"""
This function implements Newton's method of optimization.
Function 'func' have to be vectorized over its arguments.
limits describes rectangle, it is an array with n x 2 dimensions.
If optimal point 'optpoint' is provided, this function provides l2-norm precision. 
rtaparam (Return to the area parameter) let us to find appropriate solution in the direction in which
the optimization process left the given area. 
"""


def newtons(func, limits, x0, funcgrad, funchess, eps=1e-3, rtaparam=0.01, optpoint=None, maxiter = 1000):
    curr, prev, delta = x0, None, None
    niters, fcalcs, gradcalcs, hesscalcs, matrixinvers = 0, 0, 0, 0, 0
    while prev is None or np.linalg.norm(curr - prev) > eps and niters < maxiter:
        g, H = funcgrad(curr), funchess(curr)
        delta = -np.dot(np.linalg.inv(H), g)
        prev = curr
        curr = curr + delta
        #print(curr)
        while not belongs(curr, limits):
            curr -= rtaparam * delta
        niters += 1
        matrixinvers += 1
        gradcalcs += 1
        hesscalcs += 1

    calcsstats = {"iterations": niters,
                  "funccalcs": fcalcs,
                  "gradcalcs": gradcalcs,
                  "hesscalcs": hesscalcs,
                  "matrixinversions": matrixinvers}

    # while not belongs(curr, limits):
    #    curr -= rtaparam * delta

    if optpoint is None:
        return curr, func(curr), calcsstats, None
    else:
        return curr, func(curr), calcsstats, np.linalg.norm(curr - optpoint)


"""
This function implements Levenberg-Marquardt method of optimization.
Function 'func' have to be vectorized over its arguments.
limits describes rectangle, it is an array with n x 2 dimensions.
Parameter of this method regulpar might be either a number or a tuple two numbers.
In the case of tuple, this parameter specify the segment for line-search of optimal parameter.
If optimal point 'optpoint' is provided, this function provides l2-norm precision. 
rtaparam (Return to the area parameter) let us to find appropriate solution in the direction in which
the optimization process left the given area. 
"""


def levenmarq(func, limits, x0, regulpar: Sequence[int, float, tuple[float, float]],
              funcgrad, funchess, eps=1e-3, rtaparam=0.01, optpoint=None, maxiter = 1000):
    curr, prev = x0, None
    delta = None
    niters, fcalcs, gradcalcs, hesscalcs, matrixinvers = 0, 0, 0, 0, 0
    n = x0.shape[0]
    identitymatrix = np.eye(N=n, dtype=float)

    while prev is None or (np.linalg.norm(curr - prev) > eps and niters < maxiter):  # and belongs(curr, limits)):
        g, H = funcgrad(curr), funchess(curr)
        gradcalcs += 1
        hesscalcs += 1
        if isinstance(regulpar, float) or isinstance(regulpar, int):
            delta = -np.dot(np.linalg.inv(H + regulpar * identitymatrix), g)
            matrixinvers += 1
        elif isinstance(regulpar, tuple):
            linesearchres = goldenratio_method(lambda nu: func(curr-np.linalg.inv(H + nu * identitymatrix) @ g),
                                               left=regulpar[0], right=regulpar[1])
            fcalcs += linesearchres[3]
            delta = -np.linalg.inv(H + linesearchres[0] * identitymatrix) @ g
            matrixinvers += (1 + linesearchres[2])

        prev = curr
        curr = curr + delta
        while not belongs(curr, limits):
            curr -= rtaparam * delta

        niters += 1

    calcsstats = {"iterations": niters,
                  "funccalcs": fcalcs,
                  "gradcalcs": gradcalcs,
                  "hesscalcs": hesscalcs,
                  "matrixinversions": matrixinvers}

    if optpoint is None:
        return curr, func(curr), calcsstats, None
    else:
        return curr, func(curr), calcsstats, np.linalg.norm(curr - optpoint)


def gradient_descent_method(func, funcgrad, x0, limits, eps=1e-3, maxiter=1000, rtaparam=0.01, optpoint=None):
    """Gradient descent method for unconstraint optimization problem.
    given a starting point x ∈ Rⁿ,
    repeat
        1. Define direction. p := −∇f(x).
        2. Line search. Choose step length α using Armijo Line Search.
        3. Update. x := x + αp.
    until stopping criterion is satisfied.

    Parameters
    --------------------
    func : callable
        Function to be minimized.
    funcgrad : callable
        The first derivative of f.
    x0 : array
        initial value of x.
    alpha : scalar, optional
        the initial value of steplength.
    eps : float, optional
        tolerance for the norm of f_grad.
    maxiter : integer, optional
        maximum number of steps.
    """
    # initialize x, f(x), and f'(x)
    curr, fcurrval, gcurr = x0, func(x0), funcgrad(x0)
    prev, delta = None, None
    # initialize number of steps, save x and f(x)
    niters, fcalcs, gradcalcs = 0, 0, 0
    # take steps
    while prev is None or (np.linalg.norm(curr - prev) > eps and niters < maxiter):
        # determine direction
        dir = -gcurr
        prev = curr
        # calculate new x, f(x), and f'(x)
        alpha, fcurrval, _, lsfcalcs = goldenratio_method(lambda u: func(curr + u * dir))
        delta = alpha * dir
        curr = curr + delta
        print(curr)
        if not belongs(curr, limits):
            while not belongs(curr, limits):
                curr -= rtaparam * delta
            fcurrval = func(curr)
            fcalcs += 1
        gcurr = funcgrad(curr)
        gradcalcs += 1
        fcalcs += lsfcalcs
        niters += 1

    calcsstats = {"iterations": niters,
                  "funccalcs": fcalcs,
                  "gradcalcs": gradcalcs}
    if optpoint is None:
        return curr, fcurrval, calcsstats, None
    else:
        return curr, fcurrval, calcsstats, np.linalg.norm(curr - optpoint)


def conjugate_gradient_method(func, funcgrad, x0, limits, rtaparam = 0.01, eps=1e-3, maxiter=1000, optpoint=None):
    """Non-Linear Conjugate Gradient Method for optimization problem.
    Parameters
    --------------------
        func        : function to optimize
        funcgrad   : first derivative of f
        x0     : initial value of x, can be set to be any numpy vector,
        method   : method to calculate beta, can be one of the followings: FR, PR, HS, DY, HZ.
        eps      : tolerance of the difference of the gradient norm to zero
        maxiter : maximum number of iterations
    """

    # initialize some values
    curr, fcurrval, gcurr = x0, func(x0), funcgrad(x0)
    prev, delta = None, None
    fcalcs, gradcalcs = 1, 1
    p = -gcurr

    # for result tabulation
    niters = 0

    # begin iteration
    while prev is None or (np.linalg.norm(curr - prev) > eps and niters < maxiter):
        # search for step size alpha

        alpha, y_new, _, lsfcalcs = goldenratio_method(lambda u: func(curr + u * p))
        fcalcs += lsfcalcs

        # update iterate x
        delta = alpha * p
        x_new = curr + delta
        if not belongs(x_new, limits):
            while not belongs(x_new, limits):
                x_new -= rtaparam * delta
            y_new = func(x_new)
            fcalcs += 1

        gf_new = funcgrad(x_new)
        gradcalcs += 1

        # calculate beta
        # Polak-Ribiere's variant of beta
        beta = np.dot(gf_new, gf_new - gcurr) / np.dot(gcurr, gcurr)

        # update everything
        prev = curr
        curr = x_new
        fcurrval = y_new
        gcurr = gf_new
        p = -gcurr + beta * p
        niters += 1

    calcsstats = {"iterations": niters,
                  "funccalcs": fcalcs,
                  "gradcalcs": gradcalcs}

    if optpoint is None:
        return curr, fcurrval, calcsstats, None
    else:
        return curr, fcurrval, calcsstats, np.linalg.norm(curr - optpoint)

