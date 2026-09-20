import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from drpe_cipher import generate_phase_mask, encrypt_image, decrypt_image,calculate_mse , inject_key_error, apply_cropping_attack, apply_channel_noise

class DRPEApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Fourier-Domain Image Encryptor (DRPE)")
        self.root.geometry("1200x750")
        self.root.configure(bg="#121212")

        # State storage
        self.original_image: np.ndarray | None = None
        self.ciphertext: np.ndarray | None = None
        self.decrypted_image: np.ndarray | None = None
        self.r1: np.ndarray | None = None
        self.r2: np.ndarray | None = None
        self.img_dim: int = 128  # Target power-of-two dimension for speed

        # Attack states by Sayem
        self.key_error_var = tk.DoubleVar(value=0.0)
        self.crop_attack_var = tk.BooleanVar(value=False)
        self.noise_attack_var = tk.BooleanVar(value=False)

        self._setup_ui()

    def _setup_ui(self):
        # Header
        header = tk.Label(
            self.root,
            text="FOURIER-DOMAIN IMAGE ENCRYPTOR (DRPE)",
            font=("Consolas", 16, "bold"),
            fg="#00FF66",
            bg="#121212",
            pady=10
        )
        header.pack(side=tk.TOP, fill=tk.X)


        # Main Split Container
        main_frame = tk.Frame(self.root, bg="#121212")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)


        # Algorithm Selector Mode
        #added after nprs 
        self.cipher_mode = tk.StringVar(value="DRPE")
        
        mode_frame = tk.Frame(main_frame, bg="#121212")
        mode_frame.pack(fill=tk.X, pady=5)
        
        tk.Radiobutton(mode_frame, text="Standard DRPE (Linear)", variable=self.cipher_mode, value="DRPE", 
                       bg="#121212", fg="#008612", selectcolor="#2A2A2A", 
                       font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=20)
        tk.Radiobutton(mode_frame, text="Chaotic DRPE", variable=self.cipher_mode, value="CHAOS", 
                       bg="#121212", fg="#FF007F", selectcolor="#2A2A2A",
                         font=("Consolas", 10, "bold")).pack(side=tk.LEFT)

        

        # Panel 1: Input & Encryption (Left)
        left_panel = tk.LabelFrame(
            main_frame,
            text="  Input & Encryption Panel ",
            font=("Consolas", 11, "bold"),
            fg="#00E5FF",
            bg="#1E1E1E",
            padx=10,
            pady=10
        )
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        btn_load = tk.Button(
            left_panel, text="Load Image", command=self.load_image,
            bg="#2A2A2A", fg="#FFFFFF", activebackground="#3A3A3A", activeforeground="#FFFFFF",
            font=("Consolas", 10), width=18
        )
        btn_load.pack(pady=4)

        btn_gen_keys = tk.Button(
            left_panel, text="Generate Phase Keys", command=self.generate_keys,
            bg="#2A2A2A", fg="#FFFFFF", activebackground="#3A3A3A", activeforeground="#FFFFFF",
            font=("Consolas", 10), width=18
        )
        btn_gen_keys.pack(pady=4)

        btn_encrypt = tk.Button(
            left_panel, text="Encrypt Image", command=self.run_encryption,
            bg="#005F73", fg="#FFFFFF", activebackground="#0A9396", activeforeground="#FFFFFF",
            font=("Consolas", 10, "bold"), width=18
        )
        btn_encrypt.pack(pady=4)

        # new button's added here to export ciphertext
        # 
        #  
        btn_export_ciphertext = tk.Button(
            left_panel, text="Export CipherText", command=self.export_ciphertext,
            bg="#005F73", fg="#FFFFFF", activebackground="#0A9396", activeforeground="#FFFFFF",
            font=("Consolas", 10, "bold"), width=18
        )
        btn_export_ciphertext.pack(pady=4)

        btn_export_keys = tk.Button(
            left_panel, text="Export Keys (.npz)", command=self.export_keys,
            bg="#2A2A2A", fg="#FFFFFF", activebackground="#3A3A3A", activeforeground="#FFFFFF",
            font=("Consolas", 10), width=18
        )
        btn_export_keys.pack(pady=4)




        self.lbl_enc_status = tk.Label(
            left_panel, text="Status: Awaiting image...", font=("Consolas", 9),
            fg="#888888", bg="#1E1E1E"
        )
        self.lbl_enc_status.pack(pady=4)

        # Panel 2: Decryption & Testing (Right)
        right_panel = tk.LabelFrame(
            main_frame,
            text=" Decryption & Testing Panel ",
            font=("Consolas", 11, "bold"),
            fg="#FF007F",
            bg="#1E1E1E",
            padx=10,
            pady=10
        )
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)


        #buttons from importing keys and cyphertext


        btn_import_ciphertext = tk.Button(
            right_panel, text="Import CipherText", command=self.import_ciphertext,
            bg="#005F73", fg="#FFFFFF", activebackground="#0A9396", activeforeground="#FFFFFF",
            font=("Consolas", 10, "bold"), width=18
        )
        btn_import_ciphertext.pack(pady=4)
        
        btn_import_keys = tk.Button(
            right_panel, text="Import Keys (.npz)", command=self.import_keys,
            bg="#2A2A2A", fg="#FFFFFF", activebackground="#3A3A3A", activeforeground="#FFFFFF",
            font=("Consolas", 10), width=18
        )
        btn_import_keys.pack(pady=4)

        btn_decrypt = tk.Button(
            right_panel, text="Decrypt Image", command=self.run_decryption,
            bg="#9B2226", fg="#FFFFFF", activebackground="#AE2012", activeforeground="#FFFFFF",
            font=("Consolas", 10, "bold"), width=18
        )
        btn_decrypt.pack(pady=4)

        


        # --- Sayem CONTROLS ---
        # Key Sensitivity Slider
        slider_frame = tk.Frame(right_panel, bg="#1E1E1E")
        slider_frame.pack(fill=tk.X, pady=10)
        tk.Label(slider_frame, text="Key Error %:", fg="#FFFFFF", bg="#1E1E1E", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.scale_error = tk.Scale(
            slider_frame, from_=0.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL,
            variable=self.key_error_var, bg="#1E1E1E", fg="#FFFFFF", highlightthickness=0
        )
        self.scale_error.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # --- ATTACK SLIDERS ---
        # Cropping Attack Slider (0% to 100%)
        crop_frame = tk.Frame(right_panel, bg="#1E1E1E")
        crop_frame.pack(fill=tk.X, pady=5)
        tk.Label(crop_frame, text="Crop Area %:", fg="#FFFFFF", bg="#1E1E1E", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.crop_val_var = tk.DoubleVar(value=0.0)
        self.scale_crop = tk.Scale(
            crop_frame, from_=0.0, to=100.0, resolution=1.0, orient=tk.HORIZONTAL,
            variable=self.crop_val_var, bg="#1E1E1E", fg="#FFFFFF", highlightthickness=0
        )
        self.scale_crop.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Noise Attack Slider (0% to 100%)
        noise_frame = tk.Frame(right_panel, bg="#1E1E1E")
        noise_frame.pack(fill=tk.X, pady=5)
        tk.Label(noise_frame, text="Noise Power %:", fg="#FFFFFF", bg="#1E1E1E", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.noise_val_var = tk.DoubleVar(value=0.0)
        self.scale_noise = tk.Scale(
            noise_frame, from_=0.0, to=100.0, resolution=1.0, orient=tk.HORIZONTAL,
            variable=self.noise_val_var, bg="#1E1E1E", fg="#FFFFFF", highlightthickness=0
        )
        self.scale_noise.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        # -----------------------------




        self.lbl_mse = tk.Label(
            right_panel, text="Reconstruction MSE: N/A", font=("Consolas", 10, "bold"),
            fg="#FFB703", bg="#1E1E1E"
        )
        self.lbl_mse.pack(pady=6)

        # Matplotlib Display Area (Bottom)
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(10, 3.8))
        self.fig.patch.set_facecolor('#121212')

        for ax, title in zip([self.ax1, self.ax2, self.ax3], ["Original Image", "Ciphertext (Magnitude)", "Decrypted Output"]):
            ax.set_title(title, color="#FFFFFF", fontsize=10, fontname="DejaVu Sans")
            ax.axis("off")
            ax.set_facecolor('#1E1E1E')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if not file_path:
            return

        img = Image.open(file_path).convert("L")  # Convert to grayscale
        img = img.resize((self.img_dim, self.img_dim))  # Must fit Radix-2 power-of-two constraint
        self.original_image = np.array(img, dtype=float)

        self.ax1.clear()
        self.ax1.imshow(self.original_image, cmap="gray")
        self.ax1.set_title(f"Original ({self.img_dim}x{self.img_dim})", color="#FFFFFF")
        self.ax1.axis("off")
        self.canvas.draw()

        self.lbl_enc_status.config(text=f"Loaded: {os.path.basename(file_path)}", fg="#00FF66")

    def generate_keys(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first to establish dimensions.")
            return

        # 1. Define the shape FIRST, and restrict it to 2D for RGB support
        shape = self.original_image.shape[:2]

        # 2. Generate the keys based on the toggle switch
        if self.cipher_mode.get() == "DRPE":
            # Old standard random keys
            self.r1 = np.random.uniform(0, 2 * np.pi, shape)
            self.r2 = np.random.uniform(0, 2 * np.pi, shape)
        elif self.cipher_mode.get() == "CHAOS":
            from chaotic_keys import generate_chaotic_mask
            # Passwords (initial keys) for R1 and R2
            self.r1 = generate_chaotic_mask(shape, initial_key=0.34567)
            self.r2 = generate_chaotic_mask(shape, initial_key=0.76543)

        # 3. Update the UI text (The overwriting lines were removed)
        self.lbl_enc_status.config(text="Status: Phase masks R1 & R2 generated.", fg="#00FF66")

        

    def run_encryption(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return
        if self.r1 is None or self.r2 is None:
            self.generate_keys()

        
        self.ciphertext = encrypt_image(self.original_image, self.r1, self.r2)


        self.lbl_enc_status.config(text="Status: Encrypting signal...", fg="#E9D8A6")
        self.root.update_idletasks()

        self.ciphertext = encrypt_image(self.original_image, self.r1, self.r2)

        # Ciphertext is complex: visualize magnitude distribution
        cipher_display = np.abs(self.ciphertext)

        self.ax2.clear()
        self.ax2.imshow(cipher_display, cmap="inferno")
        self.ax2.set_title("Ciphertext |C(x,y)| (White Noise)", color="#FFFFFF")
        self.ax2.axis("off")
        self.canvas.draw()

        self.lbl_enc_status.config(text="Status: Encryption complete.", fg="#00FF66")



#Fully modified by Sayem


    def run_decryption(self):
        if self.ciphertext is None:
            messagebox.showwarning("Warning", "No ciphertext found. Run encryption first.")
            return

        # 1. Fetch attack parameters
        test_cipher = np.copy(self.ciphertext)
        
        # We copy r2 because injecting error into the frequency mask 
        # is what actually breaks the IDFT structure.
        test_r2 = np.copy(self.r2) 
        
        # 2. Apply Ciphertext Attacks
        crop_val = self.crop_val_var.get()
        if crop_val > 0:
            test_cipher = apply_cropping_attack(test_cipher, crop_ratio=(crop_val / 100.0))
            
        noise_val = self.noise_val_var.get()
        if noise_val > 0:
            test_cipher = apply_channel_noise(test_cipher, noise_variance=(noise_val / 100.0))
            \
            
        # 1. Apply Key Error based on the active mode
        err_val = self.key_error_var.get()
        
        if self.cipher_mode.get() == "DRPE":
            # Old linear error: sprinkles noise on the final mask
            test_r2 = inject_key_error(self.r2, err_val)
            
        elif self.cipher_mode.get() == "CHAOS":
            if err_val > 0:
                from chaotic_keys import generate_chaotic_mask
                # The original R2 password was 0.76543. 
                # We simulate a hacker guessing slightly wrong (e.g., slider at 1% = +0.00001 error)
                wrong_password = 0.76543 + (err_val * 0.00001)
                
                # Generate a completely new chaotic mask from the wrong password
                shape = self.original_image.shape[:2]
                test_r2 = generate_chaotic_mask(shape, initial_key=wrong_password)
            else:
                test_r2 = self.r2

        # # 4. Attempt Decryption (Pass self.r1 uncorrupted, pass test_r2 corrupted)
        # self.decrypted_image = decrypt_image(test_cipher, self.r1, test_r2)

        self.decrypted_image = decrypt_image(test_cipher, self.r1, test_r2)


        # 5. Calculate MSE & Update UI
        if self.original_image is not None:
            mse = calculate_mse(self.original_image, self.decrypted_image)
            self.lbl_mse.config(text=f"Reconstruction MSE: {mse:.4e}")
        else:
            self.lbl_mse.config(text="Reconstruction MSE: N/A (Receiver Mode)")

        self.ax3.clear()
        self.ax3.imshow(self.decrypted_image, cmap="gray")
        self.ax3.set_title("Decrypted Image", color="#FFFFFF")
        self.ax3.axis("off")
        self.canvas.draw()

    def export_keys(self):
        if self.r1 is None or self.r2 is None:
            messagebox.showwarning("Warning", "No keys available to export.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy Zip Archive", "*.npz")]
        )
        if save_path:
            np.savez(save_path, r1=self.r1, r2=self.r2)
            messagebox.showinfo("Export Successful", f"Keys saved to {os.path.basename(save_path)}")

    def export_ciphertext(self):
        if self.ciphertext is None:
            messagebox.showwarning("Warning", "No ciphertext to export.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy Array", "*.npy")])
        if save_path:
            np.save(save_path, self.ciphertext)
            messagebox.showinfo("Success", "Ciphertext exported successfully.")

    def import_ciphertext(self):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy Array", "*.npy")])
        if file_path:
            self.ciphertext = np.load(file_path)
            self.ax2.clear()
            self.ax2.imshow(np.abs(self.ciphertext), cmap="inferno")
            self.ax2.set_title("Ciphertext |C(x,y)|", color="#FFFFFF")
            self.ax2.axis("off")
            self.canvas.draw()
            self.lbl_mse.config(text="Status: Ciphertext Loaded.")

    def import_keys(self):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy Zip", "*.npz")])
        if file_path:
            data = np.load(file_path)
            self.r1, self.r2 = data['r1'], data['r2']
            messagebox.showinfo("Success", "Phase Keys imported successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DRPEApp(root)
    root.mainloop()