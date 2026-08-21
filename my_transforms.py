import numpy as np

def custom_fft1d_radix2(x: np.ndarray) -> np.ndarray:
    """
    1D Cooley-Tukey Radix-2 FFT. 
    Input length MUST be a power of 2.
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    
    if N <= 1:
        return x
        
    # Divide and conquer: separate even and odd indices
    even = custom_fft1d_radix2(x[0::2])
    odd = custom_fft1d_radix2(x[1::2])
    
    # Calculate the twiddle factors
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    
    # Combine
    half_N = N // 2
    return np.concatenate([
        even + factor[:half_N] * odd,
        even + factor[half_N:] * odd
    ])

def my_fft2(image: np.ndarray) -> np.ndarray:
    """
    Highly efficient 2D custom FFT.
    Applies 1D FFT to rows, then to columns.
    """
    # 1. Apply 1D FFT along rows
    row_fft = np.array([custom_fft1d_radix2(row) for row in image])
    
    # 2. Apply 1D FFT along columns (transpose, apply, transpose back)
    return np.array([custom_fft1d_radix2(col) for col in row_fft.T]).T

def my_ifft2(spectrum: np.ndarray) -> np.ndarray:
    """
    Efficient 2D custom IFFT utilizing the FFT function.
    IFFT(X) = conj(FFT(conj(X))) / N
    """
    M, N = spectrum.shape
    # Using the mathematical property that relates IFFT to FFT
    inverse_transform = np.conj(my_fft2(np.conj(spectrum)))
    return inverse_transform / (M * N)