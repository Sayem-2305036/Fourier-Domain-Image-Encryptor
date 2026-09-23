import numpy as np
from wavelet_engine import split_image, rebuild_image
from frft_engine import frft_2d
from chaotic_keys import generate_chaotic_mask
from hash_engine import derive_dynamic_keys


def _encrypt_band(band: np.ndarray, mask1: np.ndarray, mask2: np.ndarray, ax: float, ay: float) -> np.ndarray:
    step1 = band * mask1
    step2 = frft_2d(step1, ax, ay)
    step3 = step2 * mask2
    return frft_2d(step3, ax, ay)


def _decrypt_band(cipher_band: np.ndarray, mask1: np.ndarray, mask2: np.ndarray, ax: float, ay: float) -> np.ndarray:
    step1 = frft_2d(cipher_band, -ax, -ay)
    step2 = step1 * mask2
    step3 = frft_2d(step2, -ax, -ay)
    return np.real(step3 * mask1)


def encrypt_advanced(image: np.ndarray):
    # 1. Get the dynamic passwords and angles from the entire RGB image fingerprint
    s1, s2, ax, ay = derive_dynamic_keys(image)

    encrypted_channels = []

    # 2. Loop through the Red (0), Green (1), and Blue (2) layers
    for i in range(3):
        # Peel off one color layer at a time
        color_layer = image[:, :, i]
        
        # Wavelet Split
        LL, LH, HL, HH = split_image(color_layer)

        # Generate chaotic phase masks (we reuse the same s1 and s2 for all colors)
        mask1 = np.exp(1j * generate_chaotic_mask(LL.shape, s1))
        mask2 = np.exp(1j * generate_chaotic_mask(LL.shape, s2))

        # Encrypt all sub-bands for this specific color
        cipher_LL = _encrypt_band(LL, mask1, mask2, ax, ay)
        cipher_LH = _encrypt_band(LH, mask1, mask2, ax, ay)
        cipher_HL = _encrypt_band(HL, mask1, mask2, ax, ay)
        cipher_HH = _encrypt_band(HH, mask1, mask2, ax, ay)

        # Rebuild the encrypted color layer and save it
        layer_ciphertext = rebuild_image(cipher_LL, cipher_LH, cipher_HL, cipher_HH)
        encrypted_channels.append(layer_ciphertext)

    # 3. Stack the three encrypted layers back into a single 3D image block
    ciphertext = np.dstack(encrypted_channels)

    return ciphertext, (s1, s2, ax, ay)


def decrypt_advanced(ciphertext: np.ndarray, s1, s2, ax, ay, bandwidth: float):
    decrypted_channels = []

    # 1. Loop through the three encrypted color layers
    for i in range(3):
        layer_cipher = ciphertext[:, :, i]
        
        # Split the encrypted color layer back open
        cipher_LL, cipher_LH, cipher_HL, cipher_HH = split_image(layer_cipher)

        # Apply the Bandwidth Attack to this specific color
        if bandwidth < 1.0:
            from filter_engine import apply_low_pass_filter
            cipher_LL = apply_low_pass_filter(cipher_LL, bandwidth)
            cipher_LH = apply_low_pass_filter(cipher_LH, bandwidth)
            cipher_HL = apply_low_pass_filter(cipher_HL, bandwidth)
            cipher_HH = apply_low_pass_filter(cipher_HH, bandwidth)

        # Re-generate the chaotic masks (using opposite phase)
        mask1 = np.exp(-1j * generate_chaotic_mask(cipher_LL.shape, s1))
        mask2 = np.exp(-1j * generate_chaotic_mask(cipher_LL.shape, s2))

        # Reverse the FrFT DRPE on every sub-band for this color
        recovered_LL = _decrypt_band(cipher_LL, mask1, mask2, ax, ay)
        recovered_LH = _decrypt_band(cipher_LH, mask1, mask2, ax, ay)
        recovered_HL = _decrypt_band(cipher_HL, mask1, mask2, ax, ay)
        recovered_HH = _decrypt_band(cipher_HH, mask1, mask2, ax, ay)

        # Rebuild the recovered color layer
        layer_decrypted = rebuild_image(recovered_LL, recovered_LH, recovered_HL, recovered_HH)
        decrypted_channels.append(layer_decrypted)

    # 2. Stack the three recovered layers back into a full-color image
    return np.dstack(decrypted_channels)