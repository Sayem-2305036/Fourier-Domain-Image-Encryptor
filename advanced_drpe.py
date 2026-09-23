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
    # 1. Get the dynamic passwords and angles from the image fingerprint
    s1, s2, ax, ay = derive_dynamic_keys(image)

    # 2. Wavelet Split: Separate the blurry base (LL) from the sharp edges (LH, HL, HH)
    LL, LH, HL, HH = split_image(image)

    # 3. Generate chaotic phase masks using the dynamic seeds.
    # Haar's 2D DWT always halves both dimensions for every sub-band, so LL, LH,
    # HL and HH all share exactly the same shape and can reuse the same masks.
    mask1 = np.exp(1j * generate_chaotic_mask(LL.shape, s1))
    mask2 = np.exp(1j * generate_chaotic_mask(LL.shape, s2))

    # 4. Apply the FrFT DRPE to EVERY sub-band, not just the base layer.
    # Encrypting only LL and zeroing the edges threw away all the fine detail
    # permanently (the wavelet transform is lossless only if you keep all four
    # bands), which is what made correct-key decryption come back blurry.
    # Encrypting all four bands keeps the transform lossless end-to-end and
    # means a wrong key now corrupts fine detail too, not just the coarse base.
    cipher_LL = _encrypt_band(LL, mask1, mask2, ax, ay)
    cipher_LH = _encrypt_band(LH, mask1, mask2, ax, ay)
    cipher_HL = _encrypt_band(HL, mask1, mask2, ax, ay)
    cipher_HH = _encrypt_band(HH, mask1, mask2, ax, ay)

    # 5. Rebuild the final ciphertext from all four encrypted sub-bands.
    ciphertext = rebuild_image(cipher_LL, cipher_LH, cipher_HL, cipher_HH)

    return ciphertext, (s1, s2, ax, ay)


def decrypt_advanced(ciphertext: np.ndarray, s1, s2, ax, ay, bandwidth: float):
    # 1. Split the ciphertext back open into its four encrypted sub-bands
    cipher_LL, cipher_LH, cipher_HL, cipher_HH = split_image(ciphertext)

    # 2. The Bandwidth Attack: Wipe out frequencies if the slider is moved.
    # Apply it to every band so the attack still targets the whole ciphertext,
    # not just the (now no longer special) base layer.
    if bandwidth < 1.0:
        from filter_engine import apply_low_pass_filter
        cipher_LL = apply_low_pass_filter(cipher_LL, bandwidth)
        cipher_LH = apply_low_pass_filter(cipher_LH, bandwidth)
        cipher_HL = apply_low_pass_filter(cipher_HL, bandwidth)
        cipher_HH = apply_low_pass_filter(cipher_HH, bandwidth)

    # 3. Re-generate the chaotic masks (using opposite phase for decryption)
    mask1 = np.exp(-1j * generate_chaotic_mask(cipher_LL.shape, s1))
    mask2 = np.exp(-1j * generate_chaotic_mask(cipher_LL.shape, s2))

    # 4. Reverse the FrFT DRPE on every sub-band (using negative angles to
    # rotate backward), recovering full detail instead of just the base layer.
    recovered_LL = _decrypt_band(cipher_LL, mask1, mask2, ax, ay)
    recovered_LH = _decrypt_band(cipher_LH, mask1, mask2, ax, ay)
    recovered_HL = _decrypt_band(cipher_HL, mask1, mask2, ax, ay)
    recovered_HH = _decrypt_band(cipher_HH, mask1, mask2, ax, ay)

    # 5. Rebuild the final image from all four recovered sub-bands. With the
    # correct key this is now a lossless round trip (no more forced blur);
    # with a wrong key, every band -- including the fine edges -- is corrupted,
    # so the result is full-resolution noise instead of a blurry wrong guess.
    return rebuild_image(recovered_LL, recovered_LH, recovered_HL, recovered_HH)