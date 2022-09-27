import numpy as np




def difevol(func, dim, m, initpopulation = None, K=0.8, F=0.8, C = 0.9, maxiter=1000):
    currpopulation = initpopulation
    func_vec = lambda x: np.apply_along_axis(func, 0, x)
    mutants, S, U = np.zeros(shape = (dim, m)), np.zeros(shape = (dim, m)), np.zeros(shape = (dim, m))
    J = np.linspace(0, dim-1, dim).repeat(m).reshape(dim, m)
    i = 0
    while i < maxiter:
        i+=1

        for j in range(m):
            indices = np.random.choice(m-1, size = 5, replace =False )
            indices += (indices >= j)
            mutants[:,j] = currpopulation[:, indices[0]]+\
                    K*(currpopulation[:, indices[1]] - currpopulation[:, indices[2]])+\
                    F*(currpopulation[:, indices[3]] - currpopulation[:, indices[4]])
            S[:, j] = np.random.permutation(dim)

        R = np.random.rand(dim, m)
        mask1 = (R <= C) | (J != S)
        mask2 = ~mask1
        U[mask1], U[mask2] = mutants[mask1], currpopulation[mask2]
        mask3 = func_vec(currpopulation) >= func_vec(U)
        mask3 = mask3.repeat(dim).reshape(dim, m, order = "F")
        currpopulation[mask3] = U[mask3]

    finalvals = func_vec(currpopulation)
    ioptimal = np.where(finalvals == np.min(finalvals))[0]
    return currpopulation[:, ioptimal], finalvals[ioptimal]