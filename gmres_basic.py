
import numpy as np
from scipy.sparse.linalg import LinearOperator, aslinearoperator


def _as_linear_operator(A):
    if isinstance(A, LinearOperator):
        return A
    return aslinearoperator(A)


def _work_dtype(Aop, b):
    A_dtype = getattr(Aop, "dtype", None)
    if A_dtype is None:
        A_dtype = np.float64
    return np.result_type(np.asarray(b).dtype, A_dtype, np.float64)


def _init_vectors(b, x0, dtype):
    b = np.asarray(b, dtype=dtype).reshape(-1)
    if x0 is None:
        x0 = np.zeros_like(b, dtype=dtype)
    else:
        x0 = np.asarray(x0, dtype=dtype).reshape(-1)
    return b, x0


def gmres_basic(A, b, x0=None, tol=1e-8, maxit=None, reorth=False, relative_tol=True):
    Aop = _as_linear_operator(A)
    dtype = _work_dtype(Aop, b)
    b, x0 = _init_vectors(b, x0, dtype)

    n = b.size
    if maxit is None:
        maxit = n
    maxit = min(maxit, n)

    V = np.zeros((n, maxit + 1), dtype=dtype)
    H = np.zeros((maxit + 1, maxit), dtype=dtype)

    r0 = b - Aop.matvec(x0)
    beta = np.linalg.norm(r0)
    threshold = tol * beta if relative_tol else tol

    residual_history = [float(beta)]
    x = x0.copy()

    if beta <= threshold:
        return x, {
            "converged": True,
            "iters": 0,
            "residual_history": residual_history,
            "H": H[:, :0],
            "V": V[:, :1],
        }

    V[:, 0] = r0 / beta
    converged = False
    iters = 0

    for j in range(maxit):
        w = Aop.matvec(V[:, j])

        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], w)
            w = w - H[i, j] * V[:, i]

        if reorth:
            for i in range(j + 1):
                h2 = np.vdot(V[:, i], w)
                H[i, j] += h2
                w = w - h2 * V[:, i]

        H[j + 1, j] = np.linalg.norm(w)

        if H[j + 1, j] > 0:
            V[:, j + 1] = w / H[j + 1, j]

        rhs = np.zeros(j + 2, dtype=dtype)
        rhs[0] = beta

        y, *_ = np.linalg.lstsq(H[:j + 2, :j + 1], rhs, rcond=None)
        x = x0 + V[:, :j + 1] @ y

        r = b - Aop.matvec(x)
        rnorm = np.linalg.norm(r)
        residual_history.append(float(rnorm))

        iters = j + 1

        if rnorm <= threshold:
            converged = True
            break

        if H[j + 1, j] == 0:
            break

    return x, {
        "converged": converged,
        "iters": iters,
        "residual_history": residual_history,
        "H": H[:, :iters],
        "V": V[:, :iters + 1],
    }
