import numpy as np
from fourier_transforms import my_fft2, my_ifft2


def generate_phase_mask(shape: tuple[int, int], seed: int | None = None) -> np.ndarray:
    """
    Generates a statistically independent, uniformly distributed random phase mask
    R in the range [0, 2*pi).
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, size=shape)


def encrypt_image(image: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """
    Member 1 Task: Forward DRPE Encryption Pipeline.
    Ciphertext = IFFT2( FFT2( I(x, y) * exp(j * R1) ) * exp(j * R2) )
    """
    # 1. Spatial Phase Modulation
    spatial_modulated = image.astype(complex) * np.exp(1j * r1)

    # 2. Frequency Transformation
    frequency_spectrum = my_fft2(spatial_modulated)

    # 3. Frequency Phase Modulation
    freq_modulated = frequency_spectrum * np.exp(1j * r2)

    # 4. Inverse Transformation -> Complex Stationary White Noise
    ciphertext = my_ifft2(freq_modulated)
    return ciphertext


def decrypt_image(ciphertext: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """
    Member 2 Task: Reverse DRPE Decryption Pipeline.
    Recovered = | IFFT2( FFT2( Ciphertext ) * exp(-j * R2) ) * exp(-j * R1) |
    """
    # 1. Forward transform to frequency domain
    spectrum = my_fft2(ciphertext)

    # 2. Multiply by conjugate of frequency phase key exp(-j * R2)
    demodulated_freq = spectrum * np.exp(-1j * r2)

    # 3. Shift back to spatial domain
    spatial_intermediate = my_ifft2(demodulated_freq)

    # 4. Multiply by conjugate of spatial phase key exp(-j * R1)
    recovered_complex = spatial_intermediate * np.exp(-1j * r1)

    # Recover original intensity magnitude
    return np.abs(recovered_complex)


def calculate_mse(original: np.ndarray, recovered: np.ndarray) -> float:
    """
    Member 2 Task: Computes Mean Squared Error (MSE) between original and recovered signals.
    """
    return float(np.mean((original - recovered) ** 2))