import numpy as np

# def apply_low_pass_filter(image_spectrum: np.ndarray, keep_percentage: float) -> np.ndarray:
#     """
#     Simulates a bad transmission channel by wiping out high-frequency waves.
#     keep_percentage is a decimal between 0.01 (terrible signal) and 1.0 (perfect signal).
#     """
#     rows, cols = image_spectrum.shape
#     center_row, center_col = rows // 2, cols // 2
    
#     # Calculate the size of the "safe zone" box based on the percentage
#     radius_r = int(center_row * keep_percentage)
#     radius_c = int(center_col * keep_percentage)
    
#     # Create a completely blank (zeroed out) mask
#     mask = np.zeros_like(image_spectrum)
    
#     # Open a window in the center to let the slow waves pass through
#     mask[center_row - radius_r : center_row + radius_r, 
#          center_col - radius_c : center_col + radius_c] = 1
         
#     # Multiply to wipe out the outside edges (high frequencies)
#     return image_spectrum * mask



def apply_low_pass_filter(image: np.ndarray, keep_percentage: float) -> np.ndarray:
    # 1. Translate to pure frequency realm and center the slow waves
    spectrum = np.fft.fftshift(np.fft.fft2(image))

    rows, cols = spectrum.shape
    center_row, center_col = rows // 2, cols // 2

    radius_r = int(center_row * keep_percentage)
    radius_c = int(center_col * keep_percentage)

    mask = np.zeros_like(spectrum)
    # Keep the center (low frequencies / slow waves)
    mask[center_row - radius_r : center_row + radius_r, 
         center_col - radius_c : center_col + radius_c] = 1

    filtered_spectrum = spectrum * mask

    # 2. Translate back to the original spatial format
    return np.fft.ifft2(np.fft.ifftshift(filtered_spectrum))