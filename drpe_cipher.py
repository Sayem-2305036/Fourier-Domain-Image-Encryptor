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
    Forward DRPE Encryption Pipeline.
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
    Reverse DRPE Decryption Pipeline.
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
    Computes Mean Squared Error (MSE) between original and recovered signals.
    """
    return float(np.mean((original - recovered) ** 2))



#Following are implemented by Md. Rijwanullah Sayem


#Key Sensitivity (Avalanche Effect)
# To prove the system is highly sensitive to the wrong key, we need a function that 
# injects a specific percentage of error into a phase mask.
# This simulates an attacker guessing a key that is almost right, but not quite

def inject_key_error(phase_mask: np.ndarray, error_percentage: float) -> np.ndarray:
    """
    Simulates a uniform guessing error where EVERY phase value is off by a certain percentage.
    """
    if error_percentage <= 0.0:
        return phase_mask
        
    
    noise_factor = error_percentage * 10 
    
    # Generate random phase shifts between -pi and +pi
    global_noise = np.random.uniform(-np.pi, np.pi, size=phase_mask.shape)
    
    # Apply the scaled noise to the entire mask
    corrupted_mask = phase_mask + (noise_factor * global_noise)
    
    # Ensure phases wrap correctly between 0 and 2*pi
    return np.mod(corrupted_mask, 2 * np.pi)



# Signal Cropping Attack (Data Loss)
# When data is transmitted, packets get lost. We need to simulate cropping a chunk
# out of the complex ciphertext to see if the frequency-domain encryption distributes
# the data well enough to survive it.

def apply_cropping_attack(ciphertext: np.ndarray, crop_ratio: float = 0.25) -> np.ndarray:
    """
    Zeroes out a central block of the ciphertext to simulate data loss.
    """
    attacked_cipher = np.copy(ciphertext)
    M, N = attacked_cipher.shape
    
    # Calculate crop dimensions
    crop_m, crop_n = int(M * crop_ratio), int(N * crop_ratio)
    start_m, start_n = (M - crop_m) // 2, (N - crop_n) // 2
    
    # Zero out the center
    attacked_cipher[start_m:start_m+crop_m, start_n:start_n+crop_n] = 0+0j
    return attacked_cipher


# Channel Noise Simulation
# Real-world channels are noisy. 
# We will apply Additive White Gaussian Noise (AWGN) to the ciphertext.

def apply_channel_noise(ciphertext: np.ndarray, noise_variance: float = 0.1) -> np.ndarray:
    """
    Adds complex Gaussian noise to the ciphertext to simulate channel degradation.
    """
    M, N = ciphertext.shape
    # Generate complex noise
    noise_real = np.random.normal(0, np.sqrt(noise_variance), (M, N))
    noise_imag = np.random.normal(0, np.sqrt(noise_variance), (M, N))
    complex_noise = noise_real + 1j * noise_imag
    
    return ciphertext + complex_noise





