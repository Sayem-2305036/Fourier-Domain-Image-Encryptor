import hashlib
import numpy as np



"""
Tying the encryption keys directly to the physical plaintext (the image) is how modern systems
 defeat Chosen-Plaintext Attacks (CPA). If an attacker feeds our system an all-black image to 
 study how the math works, the SHA-256 engine will generate a totally different hash, resulting
 in completely different keys and rotation angles than when it encrypts the real target file. 
 The attacker learns nothing.

"""

def derive_dynamic_keys(image: np.ndarray):
    """
    Acts as a digital fingerprint machine. 
    Takes the image, creates a unique SHA-256 hash, and chops it into 4 unique numbers.
    """
    # 1. Convert the physical picture into raw computer bytes and hash it
    img_bytes = image.tobytes()
    hash_hex = hashlib.sha256(img_bytes).hexdigest()
    
    # 2. Chop the 64-character hex string into 4 equal chunks
    chunk1 = int(hash_hex[0:16], 16)
    chunk2 = int(hash_hex[16:32], 16)
    chunk3 = int(hash_hex[32:48], 16)
    chunk4 = int(hash_hex[48:64], 16)
    
    # 3. Convert the giant chunks into decimals we can use
    # Chaotic seeds must be between 0.1 and 0.9 for the pinball math to work right
    seed1 = (chunk1 / (16**16)) * 0.8 + 0.1 
    seed2 = (chunk2 / (16**16)) * 0.8 + 0.1
    
    # FrFT rotation angles can be anything, let's keep them between 0.1 and 1.9
    alpha_x = (chunk3 / (16**16)) * 1.8 + 0.1
    alpha_y = (chunk4 / (16**16)) * 1.8 + 0.1
    
    return seed1, seed2, alpha_x, alpha_y