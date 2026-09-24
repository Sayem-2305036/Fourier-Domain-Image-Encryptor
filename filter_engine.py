import numpy as np
from fourier_transforms import my_fft2, my_ifft2


def apply_low_pass_filter(image: np.ndarray, keep_percentage: float) -> np.ndarray:
    # 1. Translate to pure frequency realm and center the slow waves
    spectrum = np.fft.fftshift(my_fft2(image))

    rows, cols = spectrum.shape[:2]
    center_row, center_col = rows // 2, cols // 2

    radius_r = int(center_row * keep_percentage)
    radius_c = int(center_col * keep_percentage)

    mask = np.zeros_like(spectrum)
    # Keep the center (low frequencies / slow waves)
    mask[center_row - radius_r : center_row + radius_r, 
         center_col - radius_c : center_col + radius_c] = 1

    filtered_spectrum = spectrum * mask

    # 2. Translate back to the original spatial format
    return my_ifft2(np.fft.ifftshift(filtered_spectrum))