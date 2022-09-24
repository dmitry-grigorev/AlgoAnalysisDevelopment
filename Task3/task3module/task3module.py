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


def newtons(func, limits, x0, funcgrad, funchess, eps=1e-3, rtaparam = 0.01, optpoint=None):
    curr, prev, delta = x0, None, None
    niters, fcalcs, gradcalcs, hesscalcs, matrixinvers = 0, 0, 0, 0, 0
    while (prev is None or np.linalg.norm(curr - prev) > eps) and belongs(curr, limits):
        g, H = funcgrad(curr), funchess(curr)
        delta = -np.linalg.inv(H) @ g
        prev = curr
        curr = curr + delta
        niters += 1
        matrixinvers += 1
        gradcalcs += 1
        hesscalcs += 1

    calcsstats = {"iterations": niters,
                  "funccalcs": fcalcs,
                  "gradcalcs": gradcalcs,
                  "hesscalcs": hesscalcs,
                  "matrixinversions": matrixinvers}

    while not belongs(curr, limits):
        curr -= rtaparam*delta

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
              funcgrad, funchess, eps=1e-3, rtaparam = 0.01, optpoint=None):
    curr, prev = x0, None
    delta = None
    niters, fcalcs, gradcalcs, hesscalcs, matrixinvers = 0, 0, 0, 0, 0
    n = x0.shape[0]
    identitymatrix = np.eye(N=n, dtype=float)

    while (prev is None or np.linalg.norm(curr - prev) > eps) and belongs(curr, limits):
        g, H = funcgrad(curr), funchess(curr)
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


        prev = curr
        curr = curr + delta
        niters += 1

    while not belongs(curr, limits):
        curr -= rtaparam * delta

    calcsstats = {"iterations": niters,
                  "funccalcs": fcalcs,
                  "gradcalcs": gradcalcs,
                  "hesscalcs": hesscalcs,
                  "matrixinversions": matrixinvers}

    if optpoint is None:
        return curr, func(curr), calcsstats, None
    else:
        return curr, func(curr), calcsstats, np.linalg.norm(curr - optpoint)


def gradient_descent_method(f, funcgrad, x0, alpha=1, eps=1e-5, max_iter=1000, print_logs=False):
    """Gradient descent method for unconstraint optimization problem.
    given a starting point x ∈ Rⁿ,
    repeat
        1. Define direction. p := −∇f(x).
        2. Line search. Choose step length α using Armijo Line Search.
        3. Update. x := x + αp.
    until stopping criterion is satisfied.

    Parameters
    --------------------
    f : callable
        Function to be minimized.
    funcgrad : callable
        The first derivative of f.
    x0 : array
        initial value of x.
    alpha : scalar, optional
        the initial value of steplength.
    eps : float, optional
        tolerance for the norm of f_grad.
    max_iter : integer, optional
        maximum number of steps.

    Returns
    --------------------
    xs : array
        x in the learning path
    ys : array
        f(x) in the learning path
    """
    # initialize x, f(x), and f'(x)
    xk = x0
    fk = f(xk)
    gfk = funcgrad(xk)
    gfk_norm = np.linalg.norm(gfk)
    # initialize number of steps, save x and f(x)
    num_iter = 0
    curve_x = [xk]
    curve_y = [fk]
    if print_logs:
        print('Initial condition: y = {:.4f}, x = {} \n'.format(fk, xk))
    # take steps
    while gfk_norm > eps and num_iter < max_iter:
        # determine direction
        pk = -gfk
        # calculate new x, f(x), and f'(x)
        alpha, fk = armijo_line_search(f, xk, pk, gfk, fk, alpha0=alpha)
        xk = xk + alpha * pk
        gfk = funcgrad(xk)
        gfk_norm = np.linalg.norm(gfk)
        # increase number of steps by 1, save new x and f(x)
        num_iter += 1
        curve_x.append(xk)
        curve_y.append(fk)
        if print_logs:
            print('Iteration: {} \t y = {:.4f}, x = {}, gradient = {:.4f}'.
                  format(num_iter, fk, xk, gfk_norm))
    # print results
    if num_iter == max_iter and print_logs:
        print('\nGradient descent does not converge.')
    elif print_logs:
        print('\nSolution: \t y = {:.4f}, x = {}'.format(fk, xk))

    return np.array(curve_x), np.array(curve_y)


def conjugate_gradient_method(f, funcgrad, x0, c1=1e-4, c2=0.1, amax=None, eps=1e-5, max_iter=1000,
                              print_logs=False):
    """Non-Linear Conjugate Gradient Method for optimization problem.
    Given a starting point x ∈ ℝⁿ.
    repeat
        1. Calculate step length alpha using Wolfe Line Search.
        2. Update x_new = x + alpha * p.
        3. Calculate beta using one of available methods.
        4. Update p = -f_grad(x_new) + beta * p
    until stopping criterion is satisfied.

    Parameters
    --------------------
        f        : function to optimize
        funcgrad   : first derivative of f
        x0     : initial value of x, can be set to be any numpy vector,
        method   : method to calculate beta, can be one of the followings: FR, PR, HS, DY, HZ.
        c1       : Armijo constant
        c2       : Wolfe constant
        amax     : maximum step size
        eps      : tolerance of the difference of the gradient norm to zero
        max_iter : maximum number of iterations

    Returns
    --------------------
        curve_x  : x in the learning path
        curve_y  : f(x) in the learning path
    """

    # initialize some values
    x = x0
    y = f(x)
    gfk = funcgrad(x)
    p = -gfk
    gfk_norm = np.linalg.norm(gfk)

    # for result tabulation
    num_iter = 0
    curve_x = [x]
    curve_y = [y]
    if print_logs:
        print('Initial condition: y = {:.4f}, x = {} \n'.format(y, x))

    # begin iteration
    while gfk_norm > eps and num_iter < max_iter:
        # search for step size alpha
        alpha, y_new = wolfe_line_search(f, funcgrad, x, p, c1=c1, c2=c2, amax=amax)

        # update iterate x
        x_new = x + alpha * p
        gf_new = funcgrad(x_new)

        # calculate beta
        # TODO maybe choose only one of methods? Seems like PR method is OK

        #Polak-Ribiere variant of beta
        beta = np.dot(gf_new, gf_new - gfk) / np.dot(gfk, gfk)

        # update everything
        error = y - y_new
        x = x_new
        y = y_new
        gfk = gf_new
        p = -gfk + beta * p
        gfk_norm = np.linalg.norm(gfk)

        # result tabulation
        num_iter += 1
        curve_x.append(x)
        curve_y.append(y)
        if print_logs:
            print('Iteration: {} \t y = {:.4f}, x = {}, gradient = {:.4f}'.
                  format(num_iter, y, x, gfk_norm))

    # print results
    if num_iter == max_iter and print_logs:
        print('\nConjugate gradient descent does not converge.')
    elif print_logs:
        print('\nSolution: \t y = {:.4f}, x = {}'.format(y, x))

    return np.array(curve_x), np.array(curve_y)


def armijo_line_search(f, xk, pk, gfk, phi0, alpha0, rho=0.5, c1=1e-4):
    """Minimize over alpha, the function ``f(xₖ + αpₖ)``.
    α > 0 is assumed to be a descent direction.

    Parameters
    --------------------
    f : callable
        Function to be minimized.
    xk : array
        Current point.
    pk : array
        Search direction.
    gfk : array
        Gradient of `f` at point `xk`.
    phi0 : float
        Value of `f` at point `xk`.
    alpha0 : scalar
        Value of `alpha` at the start of the optimization.
    rho : float, optional
        Value of alpha shrinkage factor.
    c1 : float, optional
        Value to control stopping criterion.

    Returns
    --------------------
    alpha : scalar
        Value of `alpha` at the end of the optimization.
    phi : float
        Value of `f` at the new point `x_{k+1}`.
    """
    derphi0 = np.dot(gfk, pk)
    phi_a0 = f(xk + alpha0 * pk)

    while not phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        alpha0 = alpha0 * rho
        phi_a0 = f(xk + alpha0 * pk)

    return alpha0, phi_a0


def wolfe_line_search(f, f_grad, xk, pk, c1=1e-4, c2=0.9, amax=None, maxiter=10):
    """
    Find alpha that satisfies strong Wolfe conditions.
    Parameters
    ----------
    f : callable f(x)
        Objective function.
    f_grad : callable f'(x)
        Objective function gradient.
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    maxiter : int, optional
        Maximum number of iterations to perform.
    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    phi : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    """

    def phi(alpha):
        return f(xk + alpha * pk)

    def derphi(alpha):
        return np.dot(f_grad(xk + alpha * pk), pk)

    alpha_star, phi_star, derphi_star = wolfe_line_search_2(phi, derphi, c1, c2, amax, maxiter)

    if derphi_star is None:
        print('The line search algorithm did not converge')

    return alpha_star, phi_star


def wolfe_line_search_2(phi, derphi, c1=1e-4, c2=0.9, amax=None, maxiter=10):
    """
    Find alpha that satisfies strong Wolfe conditions.
    alpha > 0 is assumed to be a descent direction.
    Parameters
    ----------
    phi : callable phi(alpha)
        Objective scalar function.
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size.
    maxiter : int, optional
        Maximum number of iterations to perform.
    Returns
    -------
    alpha_star : float or None
        Best alpha, or None if the line search algorithm did not converge.
    phi_star : float
        phi at alpha_star.
    derphi_star : float or None
        derphi at alpha_star, or None if the line search algorithm
        did not converge.
    """

    phi0 = phi(0.)
    derphi0 = derphi(0.)

    alpha0 = 0
    alpha1 = 1.0

    if amax is not None:
        alpha1 = min(alpha1, amax)

    phi_a1 = phi(alpha1)
    # derphi_a1 = derphi(alpha1) evaluated below

    phi_a0 = phi0
    derphi_a0 = derphi0

    for i in range(maxiter):
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
                ((phi_a1 >= phi_a0) and (i > 1)):
            alpha_star, phi_star, derphi_star = \
                _zoom(alpha0, alpha1, phi_a0,
                      phi_a1, derphi_a0, phi, derphi,
                      phi0, derphi0, c1, c2)
            break

        derphi_a1 = derphi(alpha1)
        if abs(derphi_a1) <= -c2 * derphi0:
            alpha_star = alpha1
            phi_star = phi_a1
            derphi_star = derphi_a1
            break

        if derphi_a1 >= 0:
            alpha_star, phi_star, derphi_star = \
                _zoom(alpha1, alpha0, phi_a1,
                      phi_a0, derphi_a1, phi, derphi,
                      phi0, derphi0, c1, c2)
            break

        alpha2 = 2 * alpha1  # increase by factor of two on each iteration
        if amax is not None:
            alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1

    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None
        print('!!!The line search algorithm did not converge!!!')

    return alpha_star, phi_star, derphi_star


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
    If no minimizer can be found, return None.
    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa.
    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2):
    """
    Zoom stage of approximate line-search satisfying strong Wolfe conditions.
    """

    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here. Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval), then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is still too close to the
        # end points (or out of the interval) then use bisection

        if i > 0:
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                            a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                a_j = a_lo + 0.5 * dalpha

        # Check new value of a_j

        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= -c2 * derphi0:
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj * (a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if i > maxiter:
            # Failed to find a conforming step size
            a_star = None
            val_star = None
            valprime_star = None
            break
    return a_star, val_star, valprime_star
