from drpe_cipher import generate_phase_mask, encrypt_image, decrypt_image, calculate_mse
import numpy as np

if __name__ == "__main__":
    # Test on a dummy power-of-two matrix (e.g., 64x64)
    size = (64, 64)
    test_image = np.random.uniform(0, 255, size=size)

    # 1. Generate keys
    R1 = generate_phase_mask(size, seed=42)
    R2 = generate_phase_mask(size, seed=99)

    # 2. Forward Encryption
    ciphertext = encrypt_image(test_image, R1, R2)

    # 3. Decryption
    recovered_image = decrypt_image(ciphertext, R1, R2)

    # 4. Mathematical Validation
    mse = calculate_mse(test_image, recovered_image)
    print(f"Mean Squared Error (MSE): {mse:.2e}")
    assert mse < 1e-10, "Decryption verification failed!"
    print("Verification Successful: Perfect reconstruction achieved.")