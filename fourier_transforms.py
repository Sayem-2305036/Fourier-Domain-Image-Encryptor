import numpy as np

def custom_fft1d_radix2(x: np.ndarray) -> np.ndarray:
    """
    1D Cooley-Tukey Radix-2 Decimation-In-Time FFT.
    Input length MUST be a power of 2.
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]

    if N <= 1:
        return x

    # Divide and conquer: separate even and odd indices
    even = custom_fft1d_radix2(x[0::2])
    odd = custom_fft1d_radix2(x[1::2])

    # Twiddle factors W_N^k = exp(-2j * pi * k / N)
    factor = np.exp(-2j * np.pi * np.arange(N) / N)

    half_N = N // 2
    return np.concatenate([
        even + factor[:half_N] * odd,
        even + factor[half_N:] * odd
    ])

def my_fft2(image: np.ndarray) -> np.ndarray:
    """
    2D FFT computed via separable 1D row-column operations.
    Dimensions must be powers of 2.
    """
    # 1. 1D FFT along rows
    row_fft = np.array([custom_fft1d_radix2(row) for row in image])
    # 2. 1D FFT along columns
    return np.array([custom_fft1d_radix2(col) for col in row_fft.T]).T

def my_ifft2(spectrum: np.ndarray) -> np.ndarray:
    """
    2D IFFT leveraging conjugate symmetry property:
    IFFT2(X) = (1 / (M * N)) * conj(FFT2(conj(X)))
    """
    M, N = spectrum.shape
    inverse_transform = np.conj(my_fft2(np.conj(spectrum)))
    return inverse_transform / (M * N)