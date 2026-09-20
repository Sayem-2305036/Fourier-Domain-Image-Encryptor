import numpy as np

def generate_chaotic_mask(shape: tuple, initial_key: float) -> np.ndarray:
    """
    Creates a chaotic phase mask. 
    initial_key must be a decimal between 0.0 and 1.0 (e.g., 0.12345).
    """
    total_pixels = np.prod(shape)
    chaos_sequence = np.zeros(total_pixels)
    
    # 1. Set the starting position (Our Password)
    x = initial_key 
    
    # The 'r' value controls the chaos. 3.99 means maximum unpredictable chaos.
    r = 3.99 
    
    # 2. Run the pinball machine loop!
    for i in range(total_pixels):
        x = r * x * (1 - x)
        chaos_sequence[i] = x
        
    # 3. Reshape the long sequence back into a 2D image shape
    chaos_matrix = chaos_sequence.reshape(shape)
    
    # 4. Scale the decimals (0 to 1) into phase angles (0 to 2*pi)
    phase_mask = chaos_matrix * 2 * np.pi
    
    return phase_mask