import numpy as np
from fourier_transforms import my_fft2, my_ifft2


def generate_phase_mask(shape: tuple[int, int], seed: int | None = None) -> np.ndarray:
    """
    Generates a statistically independent, uniformly distributed random phase mask
    R in the range [0, 2*pi).
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 2 * np.pi, size=shape)


def encrypt_image(image: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Encrypts an RGB image using standard linear DRPE with proper phase shifts.
    """
    encrypted_channels = []
    
    # CRITICAL FIX: Convert flat random numbers into complex wave angles (Phase Masks)
    phase_mask1 = np.exp(1j * mask1)
    phase_mask2 = np.exp(1j * mask2)

    # Loop through the Red (0), Green (1), and Blue (2) layers
    for i in range(3):
        channel = image[:, :, i]

        # Apply the complex phase masks
        step1 = channel * phase_mask1
        step2 = np.fft.fft2(step1)
        step3 = step2 * phase_mask2
        cipher_channel = np.fft.ifft2(step3)

        encrypted_channels.append(cipher_channel)

    # Glue the three encrypted layers back together into a 3D block
    return np.dstack(encrypted_channels)

def decrypt_image(ciphertext: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    """
    Decrypts an RGB image using standard linear DRPE.
    """
    decrypted_channels = []
    
    # CRITICAL FIX: Reverse the wave angles by using negative phases
    inv_phase_mask1 = np.exp(-1j * mask1)
    inv_phase_mask2 = np.exp(-1j * mask2)

    # Loop through the three encrypted color layers
    for i in range(3):
        cipher_channel = ciphertext[:, :, i]

        # Reverse the standard DRPE math
        step1 = np.fft.fft2(cipher_channel)
        step2 = step1 * inv_phase_mask2 
        step3 = np.fft.ifft2(step2)
        
        # Multiply by opposite phase of mask1 and extract physical brightness
        recovered_channel = np.abs(step3 * inv_phase_mask1)

        decrypted_channels.append(recovered_channel)

    # Glue the three recovered layers back together
    return np.dstack(decrypted_channels)


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
        
    
    noise_factor = error_percentage / 100 
    
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

def apply_cropping_attack(ciphertext: np.ndarray, crop_ratio: float = 0.0) -> np.ndarray:
    """
    Zeroes out a central block of the ciphertext to simulate data loss.
    """
    attacked_cipher = np.copy(ciphertext)
    M, N = attacked_cipher.shape[:2]
    
    # Calculate crop dimensions
    crop_m, crop_n = int(M * crop_ratio), int(N * crop_ratio)
    start_m, start_n = (M - crop_m) // 2, (N - crop_n) // 2
    
    # Zero out the center
    attacked_cipher[start_m:start_m+crop_m, start_n:start_n+crop_n] = 0+0j
    return attacked_cipher


# Channel Noise Simulation
# Real-world channels are noisy. 
# We will apply Additive White Gaussian Noise (AWGN) to the ciphertext.

def apply_channel_noise(ciphertext: np.ndarray, noise_variance: float = 0.0) -> np.ndarray:
    """
    Adds complex Gaussian noise scaled relative to the ciphertext's signal power.
    noise_variance of 1.0 means Noise Power == Signal Power.
    """
    if noise_variance <= 0.0:
        return ciphertext
        
    # 1. Calculate the true average power (energy) of the ciphertext signal
    signal_power = float(np.mean(np.abs(ciphertext)**2))
    
    # 2. Scale the noise power based on the slider ratio
    target_noise_power = signal_power * noise_variance
    
    # 3. Generate complex noise (split power between real and imaginary parts)
    noise_real = np.random.normal(0, np.sqrt(target_noise_power / 2), ciphertext.shape)
    noise_imag = np.random.normal(0, np.sqrt(target_noise_power / 2), ciphertext.shape)
    
    complex_noise = noise_real + 1j * noise_imag
    
    return ciphertext + complex_noise





