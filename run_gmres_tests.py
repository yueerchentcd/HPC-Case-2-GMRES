
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.io import mmread

from gmres_basic import gmres_basic
from gmres_givens import gmres_givens
from gmres_givens_lazy import gmres_givens_lazy


def load_matrix_market(path):
    A = mmread(path)
    if sp.issparse(A):
        A = A.tocsr()
    else:
        A = sp.csr_matrix(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{path} is not square: {A.shape}")
    return A


def run_one_case(case_name, A, tol=1e-8, maxit=None):
    n = A.shape[0]
    x_true = np.ones(n)
    b = A @ x_true
    x0 = np.zeros(n)

    solvers = [
        ("basic", gmres_basic),
        ("givens", gmres_givens),
        ("givens_lazy", gmres_givens_lazy),
    ]

    print(f"\n=== {case_name} ===")
    print(f"shape = {A.shape}, nnz = {A.nnz}")

    histories = {}

    for label, solver in solvers:
        x, info = solver(A, b, x0=x0, tol=tol, maxit=maxit, reorth=False)

        if x is None:
            rel_res = np.nan
            rel_err = np.nan
            status = "did not converge before stopping"
        else:
            rel_res = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
            rel_err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
            status = "converged" if info["converged"] else "stopped without convergence"

        print(
            f"{label:12s} | {status:30s} | "
            f"iters = {info['iters']:4d} | "
            f"rel_res = {rel_res:.3e} | rel_err = {rel_err:.3e}"
        )

        histories[label] = info["residual_history"]

    plt.figure(figsize=(7, 5))
    for label, hist in histories.items():
        plt.semilogy(hist, marker="o", markersize=3, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Residual norm / estimate")
    plt.title(f"GMRES convergence on {case_name}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    small_path = Path("small_test.mtx")
    large_path = Path("large_test.mtx")

    if not small_path.exists():
        raise FileNotFoundError("small_test.mtx not found")
    if not large_path.exists():
        raise FileNotFoundError("large_test.mtx not found")

    A_small = load_matrix_market(small_path)
    A_large = load_matrix_market(large_path)

    run_one_case(
        "small SuiteSparse test",
        A_small,
        tol=1e-8,
        maxit=A_small.shape[0],
    )

    run_one_case(
        "large SuiteSparse test",
        A_large,
        tol=1e-8,
        maxit=min(A_large.shape[0], 300),
    )


if __name__ == "__main__":
    main()
