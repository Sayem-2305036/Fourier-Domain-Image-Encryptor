import numpy as np
from fourier_transforms import my_fft2, my_ifft2

def encrypt_nprs(image: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """
    Non-Linear Encryption: Converts to frequency domain and TRUNCATES the phase.
    """
    # 1. Modulate with spatial key
    spatial_modulated = image.astype(complex) * np.exp(1j * r1)
    
    # 2. Shift to frequency domain
    frequency_spectrum = my_fft2(spatial_modulated)
    
    # 3. THE NON-LINEAR TRAP: Keep only the magnitude, permanently delete the phase
    ciphertext = np.abs(frequency_spectrum) 
    
    return ciphertext



def decrypt_nprs(ciphertext: np.ndarray, r1: np.ndarray, iterations: int = 50) -> np.ndarray:
    """
    Iterative Decryption (Gerchberg-Saxton Loop) for Phase Retrieval.
    """
    # 1. Initialize with a random phase to give the solver unbiased degrees of freedom
    current_phase = np.random.uniform(-np.pi, np.pi, ciphertext.shape)
    
    for _ in range(iterations):
        # Combine known ciphertext magnitude with current frequency phase guess
        freq_guess = ciphertext * np.exp(1j * current_phase)
        
        # Shift back to spatial domain
        spatial_guess = my_ifft2(freq_guess)
        
        # Remove the spatial key to uncover the raw image guess
        raw_image_guess = spatial_guess * np.exp(-1j * r1)
        
        # THE FIX: Force the image to be real-valued and non-negative.
        # This properly constrains the phase without erasing the solver's progress.
        image_constrained = np.maximum(0, np.real(raw_image_guess))
        
        # Re-apply the spatial key
        spatial_corrected = image_constrained * np.exp(1j * r1)
        
        # Shift back to frequency domain to update the phase guess
        freq_new = my_fft2(spatial_corrected)
        current_phase = np.angle(freq_new)
        
    # Final reconstruction step
    final_spatial = my_ifft2(ciphertext * np.exp(1j * current_phase))
    final_image = final_spatial * np.exp(-1j * r1)
    
    return np.maximum(0, np.real(final_image))



# We are only using R1 here to keep the iterative loop fast and 
# stable for our project. Adding R2 requires a double-loop which can take minutes to compute
# without a dedicated graphics card.