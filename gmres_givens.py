
import numpy as np
from scipy.linalg import solve_triangular
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


def _complex_givens(a, b):
    if abs(b) == 0:
        return 1.0, 0.0
    if abs(a) == 0:
        return 0.0, 1.0

    scale = abs(a) + abs(b)
    norm = scale * np.sqrt(abs(a / scale) ** 2 + abs(b / scale) ** 2)
    alpha = a / abs(a)

    c = abs(a) / norm
    s = np.conjugate(alpha) * b / norm
    return c, s


def _apply_givens(c, s, x, y):
    x_new = c * x + s * y
    y_new = -np.conjugate(s) * x + c * y
    return x_new, y_new


def gmres_givens(A, b, x0=None, tol=1e-8, maxit=None, reorth=False, relative_tol=True):
    Aop = _as_linear_operator(A)
    dtype = _work_dtype(Aop, b)
    b, x0 = _init_vectors(b, x0, dtype)

    n = b.size
    if maxit is None:
        maxit = n
    maxit = min(maxit, n)

    V = np.zeros((n, maxit + 1), dtype=dtype)
    H = np.zeros((maxit + 1, maxit), dtype=dtype)

    cs = np.zeros(maxit, dtype=dtype)
    sn = np.zeros(maxit, dtype=dtype)
    g = np.zeros(maxit + 1, dtype=dtype)

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
            "g": g[:1],
            "cs": cs[:0],
            "sn": sn[:0],
        }

    V[:, 0] = r0 / beta
    g[0] = beta

    converged = False
    iters = 0

    for j in range(maxit):
        w = Aop.matvec(V[:, j])

        # Arnoldi
        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], w)
            w = w - H[i, j] * V[:, i]

        if reorth:
            for i in range(j + 1):
                h2 = np.vdot(V[:, i], w)
                H[i, j] += h2
                w = w - h2 * V[:, i]

        h_subdiag = np.linalg.norm(w)
        H[j + 1, j] = h_subdiag

        if h_subdiag > 0:
            V[:, j + 1] = w / h_subdiag

        # Apply previous Givens rotations
        for i in range(j):
            H[i, j], H[i + 1, j] = _apply_givens(cs[i], sn[i], H[i, j], H[i + 1, j])

        # Apply new Givens rotation
        cs[j], sn[j] = _complex_givens(H[j, j], H[j + 1, j])
        H[j, j], H[j + 1, j] = _apply_givens(cs[j], sn[j], H[j, j], H[j + 1, j])

        # Update RHS
        g[j], g[j + 1] = _apply_givens(cs[j], sn[j], g[j], g[j + 1])

        residual_est = abs(g[j + 1])
        residual_history.append(float(residual_est))

        y = solve_triangular(H[:j + 1, :j + 1], g[:j + 1], lower=False)
        x = x0 + V[:, :j + 1] @ y

        iters = j + 1

        if residual_est <= threshold:
            converged = True
            break

        # Lucky breakdown must use the pre-rotation subdiagonal
        if h_subdiag == 0:
            break

    return x, {
        "converged": converged,
        "iters": iters,
        "residual_history": residual_history,
        "H": H[:, :iters],
        "V": V[:, :iters + 1],
        "g": g[:iters + 1],
        "cs": cs[:iters],
        "sn": sn[:iters],
    }
