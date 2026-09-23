import numpy as np
import pywt

def split_image(image: np.ndarray):
    """
    Splits the image into a blurry base and three edge layers.
    We use the 'haar' wavelet, which is the simplest and fastest mathematical strainer.
    """
    # dwt2 stands for 2D Discrete Wavelet Transform
    coeffs2 = pywt.dwt2(image, 'haar')
    
    # LL = Blurry base. LH, HL, HH = Horizontal, Vertical, and Diagonal edges
    LL, (LH, HL, HH) = coeffs2
    
    return LL, LH, HL, HH

def rebuild_image(LL, LH, HL, HH) -> np.ndarray:
    """
    Stitches the four layers back together into the full image.
    """
    coeffs2 = LL, (LH, HL, HH)
    
    # idwt2 stands for Inverse 2D Discrete Wavelet Transform
    recovered_image = pywt.idwt2(coeffs2, 'haar')
    
    return recovered_image