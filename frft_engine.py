import numpy as np
from functools import lru_cache


@lru_cache(maxsize=None)
def _build_frft_eigenbasis(N: int):
    """
    Builds the eigenvector basis used to define the Discrete Fractional Fourier
    Transform (DFrFT) as a genuine fractional *power* of the unitary DFT matrix.

    The N-point unitary DFT matrix F has exactly four eigenvalues,
    {1, -i, -1, i} = exp(-i*k*pi/2) for k = 0, 1, 2, 3, each with its own
    eigenspace. Any two Hermitian matrices built from polynomials in F commute
    with F, so their eigenvectors are also eigenvectors of F; combining two
    such matrices (S = F + F^-1 and T = i(F - F^-1)) gives each of the four
    eigenspaces a distinct combined eigenvalue, which lets us sort every
    eigenvector into its correct k-class unambiguously and with an orthonormal
    basis (via eigh, since the combined matrix is Hermitian).

    Once every eigenvector v_k is tagged with its class k, the alpha-th
    fractional power of F is defined as

        F_alpha = sum_k exp(-i * alpha * k * pi/2) * v_k v_k^H

    This is unitary for every real alpha, satisfies F_0 = I, F_1 = F (the
    ordinary DFT), F_3 = F^-1, and critically F_a . F_b = F_(a+b) for *any*
    real a, b -- so F_(-a) is guaranteed to be the exact inverse of F_a. This
    is the standard "eigen-decomposition" construction of the DFrFT (Pei &
    Yeh 1997 / Santhanam & McClellan 1996) and is what should be used instead
    of a naive two-chirp-multiply-plus-plain-FFT shortcut, which only equals
    the correct kernel when the rotation angle is exactly a multiple of pi/2.
    """
    n = np.arange(N)
    F = np.exp(-2j * np.pi * np.outer(n, n) / N) / np.sqrt(N)
    F_inv = F.conj().T  # unitary, so F^-1 = F^H

    S = F + F_inv               # eigenvalues 2*cos(k*pi/2): separates {0,2} from {1,3}
    T = 1j * (F - F_inv)        # eigenvalues -2*sin(k*pi/2): separates {1,3} from each other
    M = S + 0.5 * T             # combined operator: all four classes get distinct eigenvalues
    M = (M + M.conj().T) / 2    # enforce exact Hermitian symmetry (removes fp round-off)

    eigvals, V = np.linalg.eigh(M)

    targets = np.array([2.0, 1.0, -2.0, -1.0])  # combined-eigenvalue signature of k=0,1,2,3
    k_of = np.array([int(np.argmin(np.abs(targets - ev))) for ev in eigvals])

    return V, k_of


def frft_1d(signal: np.ndarray, alpha: float) -> np.ndarray:
    """
    1D Discrete Fractional Fourier Transform, defined as the alpha-th
    fractional power of the unitary DFT matrix (see _build_frft_eigenbasis).
    Exactly invertible via frft_1d(frft_1d(x, alpha), -alpha) == x for any
    real alpha, and additive: frft_1d(frft_1d(x, a), b) == frft_1d(x, a + b).
    """
    signal = np.asarray(signal, dtype=complex)
    N = signal.shape[0]
    V, k_of = _build_frft_eigenbasis(N)

    phase = np.exp(-1j * alpha * k_of * np.pi / 2)
    # (V * phase) @ (V^H @ signal), computed without forming the full N x N matrix
    coeffs = V.conj().T @ signal
    return V @ (phase * coeffs)


def frft_2d(image: np.ndarray, alpha_x: float, alpha_y: float) -> np.ndarray:
    """
    Applies the FrFT to a 2D image by rotating the rows, then the columns.
    """
    rows, cols = image.shape
    intermediate = np.zeros_like(image, dtype=complex)
    final_result = np.zeros_like(image, dtype=complex)

    # Process horizontal waves
    for i in range(rows):
        intermediate[i, :] = frft_1d(image[i, :], alpha_x)

    # Process vertical waves
    for j in range(cols):
        final_result[:, j] = frft_1d(intermediate[:, j], alpha_y)

    return final_result