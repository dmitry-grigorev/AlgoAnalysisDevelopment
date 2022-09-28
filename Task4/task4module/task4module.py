import copy

import numpy as np


def difevol(func, dim, m, initpopulation=None, eps=1e-3, K=0.8, F=0.8, C=0.9, maxiter=1000, maxbeststayiters=10,
            optpoint=None):
    currpopulation = initpopulation
    func_vec = lambda x: np.apply_along_axis(func, 0, x)
    currvals = func_vec(currpopulation)
    mutants, S, U = np.zeros(shape=(dim, m)), np.zeros(shape=(dim, m)), np.zeros(shape=(dim, m))
    J = np.linspace(0, dim - 1, dim).repeat(m).reshape(dim, m)
    i, ibeststay = 0, 0
    fcalcs = m
    currbest, prevbest = currpopulation[:, 0], None
    while prevbest is None or (i < maxiter and ibeststay < maxbeststayiters):
        i += 1
        prevbest = currbest
        for j in range(m):
            indices = np.random.choice(m - 1, size=5, replace=False)
            indices += (indices >= j)
            mutants[:, j] = currpopulation[:, indices[0]] + \
                            K * (currpopulation[:, indices[1]] - currpopulation[:, indices[2]]) + \
                            F * (currpopulation[:, indices[3]] - currpopulation[:, indices[4]])
            S[:, j] = np.random.permutation(dim)

        R = np.random.rand(dim, m)
        mask1 = (R <= C) | (J != S)
        mask2 = ~mask1
        U[mask1], U[mask2] = mutants[mask1], currpopulation[mask2]
        Uvals = func_vec(U)
        mask3 = currvals >= Uvals
        fcalcs += m
        currvals[mask3] = Uvals[mask3]
        mask3 = mask3.repeat(dim).reshape(dim, m, order="F")
        currpopulation[mask3] = U[mask3]
        curriopt = np.where(currvals == np.min(currvals))[0]
        currbest = currpopulation[:, curriopt]

        if np.linalg.norm(currbest - prevbest) > eps:
            ibeststay = 0
        else:
            ibeststay += 1

    calcsstats = {"iterations": i,
                  "funccalcs": fcalcs}

    finalvals = func_vec(currpopulation)
    ioptimal = np.where(finalvals == np.min(finalvals))[0]
    if optpoint is None:
        return currpopulation[:, ioptimal], finalvals[ioptimal], calcsstats, None
    else:
        return currpopulation[:, ioptimal], finalvals[ioptimal], calcsstats, \
               np.linalg.norm(currpopulation[:, ioptimal] - optpoint)


def safortsp(paths: np.ndarray, initpath, Tinit=1, alpha=0.9):
    n = paths.shape[0]
    # random initial tour
    X, Xlen = initpath, distance(paths, initpath)
    T = Tinit
    t = 0
    # X.append(X[0])
    while T > 0.0001:
        t += 1
        Xnew = reproduce_new_path(X)
        Xnewlen = distance(paths, Xnew)
        diff = Xnewlen - Xlen
        if diff < 0 or np.exp(-diff / T) > np.random.rand():
            X, Xlen = Xnew, Xnewlen
        if t % 5 == 0:
            T *= alpha

    return X, Xlen


def distance(M, X):
    d = 0
    n = len(X)
    for i in range(n):
        d += M[X[i]][X[(i + 1) % n]]
    return d


def reproduce_new_path(X: np.ndarray):
    X_new = copy.deepcopy(X)
    n = X.shape[0]
    i = np.random.randint(low=0, high=n - 2)
    j = np.random.randint(low=i + 1, high=n - 1)
    if np.random.rand() < 0.5:
        X_new[i:(j + 1)] = np.flip(X[i:(j + 1)])
    else:
        X_new = np.delete(X, np.s_[i:(j + 1)])
        n1 = X_new.shape[0]
        if n1 > 1:
            k = np.random.choice(n1 - 1) + 1
            X_new = np.insert(X_new, k, X[i:(j + 1)])
        else:
            X_new = np.append(X_new, X[i:(j + 1)])
    return X_new
