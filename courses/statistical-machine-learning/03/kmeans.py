import numpy as np
# -------------------------------------------------
# K-means
def classify(x, means, max_iter):
    N = np.size(x, 0)
    M = np.size(x, 1)
    c = np.zeros(shape=(N, M))
    i_iter = 0
    sq_dists = np.zeros(shape=(N, M))

    while i_iter < max_iter:
        i_iter = i_iter + 1
        old_means = means
        for i in range(0, N):
            for j in range(0, M):
                d = np.sqrt(np.sum(np.square(x[i, j] - means), 1))
                d2 = np.square(d)
                index = np.argmin(d2)
                minVal = d2[index]
                sq_dists[i, j] = minVal
                c[i, j] = index

        for i in range(0, len(means)):
            members = x[c == i]
            if len(members) == 0:
                means[i] = 0
            else:
                means[i] = np.mean(members, 0)

        if old_means == means:
            break
    return c
