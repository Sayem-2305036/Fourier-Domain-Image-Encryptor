import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from drpe_cipher import generate_phase_mask, encrypt_image, decrypt_image,calculate_mse , inject_key_error, apply_cropping_attack, apply_channel_noise
from fourier_transforms import my_fft2, my_ifft2

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
        self.advanced_keys = None  # ADDED: Storage for the dynamic hashing keys
        self.img_dim: int = 128  # Target power-of-two dimension for speed

        # Attack states by Sayem
        self.key_error_var = tk.DoubleVar(value=0.0)
        self.crop_attack_var = tk.BooleanVar(value=False)
        self.noise_attack_var = tk.BooleanVar(value=False)

        self._setup_ui()

    def _setup_ui(self):
        # Top header with title
        title_frame = tk.Frame(self.root, bg="#121212", height=30)
        title_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        title_frame.pack_propagate(False)
        
        header = tk.Label(
            title_frame,
            text="FOURIER-DOMAIN IMAGE ENCRYPTOR (DRPE)",
            font=("Consolas", 16, "bold"),
            fg="#00FF66",
            bg="#121212",
        )
        header.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Top right - Buttons (Reset then Exit)
        button_frame = tk.Frame(self.root, bg="#121212", height=40)
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        button_frame.pack_propagate(False)
        
        # Spacer on left
        spacer = tk.Frame(button_frame, bg="#121212")
        spacer.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Right side - Buttons (Reset then Exit)
        btn_reset = tk.Button(
            button_frame, text="Reset", command=self.reset_all,
            bg="#FF6B35", fg="#FFFFFF", activebackground="#FF8C5A", activeforeground="#FFFFFF",
            font=("Consolas", 9, "bold"), width=8
        )
        btn_reset.pack(side=tk.RIGHT, padx=2)
        
        btn_exit = tk.Button(
            button_frame, text="Exit", command=self.exit_app,
            bg="#DC2F02", fg="#FFFFFF", activebackground="#F77F00", activeforeground="#FFFFFF",
            font=("Consolas", 9, "bold"), width=8
        )
        btn_exit.pack(side=tk.RIGHT, padx=2)


        # Main Split Container
        main_frame = tk.Frame(self.root, bg="#121212")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)


        # Algorithm Selector Mode
        #added after nprs 
        self.cipher_mode = tk.StringVar(value="DRPE")
        self.cipher_mode.trace_add("write", lambda *args: self._update_key_button_labels())
        
        mode_frame = tk.Frame(main_frame, bg="#121212")
        mode_frame.pack(fill=tk.X, pady=5)
        
        tk.Radiobutton(mode_frame, text="Standard DRPE (Linear)", variable=self.cipher_mode, value="DRPE", 
                       bg="#121212", fg="#008612", selectcolor="#2A2A2A", 
                       font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=20)
        tk.Radiobutton(mode_frame, text="Advanced DRPE", variable=self.cipher_mode, value="ADVANCED", 
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
        self.btn_export_keys = btn_export_keys  # Keep reference to update label

        # Status message - below Export Keys button
        self.lbl_global_status = tk.Label(
            left_panel, text="Status: Awaiting image...", 
            font=("Consolas", 9), fg="#888888", bg="#1E1E1E",
            width=35, anchor="w", justify="left"
        )
        self.lbl_global_status.pack(pady=6, fill=tk.X)




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
        self.btn_import_keys = btn_import_keys  # Keep reference to update label

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
        self.lbl_key_error = tk.Label(slider_frame, text="Key Error %:", fg="#FFFFFF", bg="#1E1E1E", font=("Consolas", 9))
        self.lbl_key_error.pack(side=tk.LEFT)
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

        # Bandwidth / Low-Pass Filter Slider
        self.bandwidth_var = tk.DoubleVar(value=1.0)
        
        bw_frame = tk.Frame(right_panel, bg="#121212")
        bw_frame.pack(fill=tk.X, pady=5)
        tk.Label(bw_frame, text="Bandwidth % (1.0 = Perfect):", bg="#121212", fg="#00FF66",
                  width=25, anchor="e").pack(side=tk.LEFT)
        tk.Scale(bw_frame, variable=self.bandwidth_var, from_=0.01, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, 
                 bg="#121212", fg="white", highlightthickness=0).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)




        self.lbl_mse = tk.Label(
            right_panel, text="Reconstruction MSE: N/A", font=("Consolas", 10, "bold"),
            fg="#FFB703", bg="#1E1E1E", width=35, anchor="w", justify="left"
        )
        self.lbl_mse.pack(pady=6, fill=tk.X)

        # Matplotlib Display Area (Bottom) - with fixed dimensions
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(14, 4))
        self.fig.patch.set_facecolor('#121212')
        self.fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.1, wspace=0.3)

        for ax, title in zip([self.ax1, self.ax2, self.ax3], ["Original Image", "Ciphertext (Magnitude)", "Decrypted Output"]):
            ax.set_title(title, color="#FFFFFF", fontsize=10, fontname="DejaVu Sans")
            ax.axis("off")
            ax.set_facecolor('#1E1E1E')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _update_key_button_labels(self):
        """Update button labels based on cipher mode."""
        mode = self.cipher_mode.get()
        if mode == "DRPE":
            self.btn_export_keys.config(text="Export Keys (.npz)")
            self.btn_import_keys.config(text="Import Keys (.npz)")
            self.lbl_key_error.config(text="Key Error %:")
        else:  # ADVANCED
            self.btn_export_keys.config(text="Export Seeds (.npz)")
            self.btn_import_keys.config(text="Import Seeds (.npz)")
            self.lbl_key_error.config(text="Seed Error %:")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if not file_path:
            return

        # Convert to RGB instead of L (grayscale)
        img = Image.open(file_path).convert("RGB")
        img = img.resize((self.img_dim, self.img_dim))
        self.original_image = np.array(img, dtype=float)

        self.ax1.clear()
        # Matplotlib needs RGB floats to be scaled between 0 and 1
        display_img = self.original_image / 255.0
        
        # We removed cmap="gray" so it draws in full color
        self.ax1.imshow(display_img)
        self.ax1.set_title(f"Original ({self.img_dim}x{self.img_dim})", color="#FFFFFF")
        self.ax1.axis("off")
        self.canvas.draw_idle()

        self.lbl_global_status.config(text=f"Loaded: {os.path.basename(file_path)}", fg="#00FF66")

    def generate_keys(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first to establish dimensions.")
            return

        shape = self.original_image.shape[:2]

        if self.cipher_mode.get() == "DRPE":
            self.r1 = np.random.uniform(0, 2 * np.pi, shape)
            self.r2 = np.random.uniform(0, 2 * np.pi, shape)
            self.lbl_global_status.config(text="Status: Standard Phase masks R1 & R2 generated.", 
                                       fg="#00FF66")
            
        elif self.cipher_mode.get() == "ADVANCED":
            self.r1, self.r2 = None, None
            self.lbl_global_status.config(text="Status: Keys are created automatically during encryption.",
                                        fg="#00FF66")

        

    def run_encryption(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return

        #Don't know if I should keep this part or not
        # if self.r1 is None or self.r2 is None:
        #     self.generate_keys()

        
        mode = self.cipher_mode.get()

        if mode == "DRPE":
            if  (self.r1 is None or self.r2 is None):
                messagebox.showwarning("Warning", "Please generate phase keys first.")
                return
            # Run the old linear encryption
            
            self.ciphertext = encrypt_image(self.original_image, self.r1, self.r2)
            self.lbl_global_status.config(text="Status: Standard DRPE Encryption complete.", fg="#00FF66")
            
        elif mode == "ADVANCED":
            from advanced_drpe import encrypt_advanced
            # Run the new pipeline and save the dynamic keys it creates
            self.ciphertext, self.advanced_keys = encrypt_advanced(self.original_image)
            self.lbl_global_status.config(text="Status: Advanced DRPE Encryption complete.", fg="#00FF66")

        # # Ciphertext is complex: visualize magnitude distribution
        # cipher_display = np.abs(self.ciphertext)

        # self.ax2.clear()
        # self.ax2.imshow(cipher_display, cmap="inferno")
        # self.ax2.set_title("Ciphertext |C(x,y)| (White Noise)", color="#FFFFFF")
        # self.ax2.axis("off")
        # self.canvas.draw_idle()

        # Extract magnitude and scale it for the screen
        cipher_display = np.abs(self.ciphertext)
        cipher_display = cipher_display / np.max(cipher_display)
        
        self.ax2.clear()
        self.ax2.imshow(cipher_display) # Removed cmap="inferno"
        self.ax2.set_title("Ciphertext |C(x,y)| (White Noise)", color="#FFFFFF")
        self.lbl_global_status.config(text="Status: Encryption complete.", fg="#00FF66")
        self.canvas.draw_idle()


#Fully modified by Sayem


    def run_decryption(self):
        if self.ciphertext is None:
            messagebox.showwarning("Warning", "No ciphertext found. Run encryption first.")
            return

        # 1. Fetch attack parameters
        test_cipher = np.copy(self.ciphertext)
        
        # We copy r2 because injecting error into the frequency mask 
        # is what actually breaks the IDFT structure.
        # test_r2 = np.copy(self.r2)
        test_r2 = np.copy(self.r2) if self.r2 is not None else None 
        
        # 2. Apply Ciphertext Attacks
        crop_val = self.crop_val_var.get()
        if crop_val > 0:
            test_cipher = apply_cropping_attack(test_cipher, crop_ratio=(crop_val / 100.0))
            
        noise_val = self.noise_val_var.get()
        if noise_val > 0:
            test_cipher = apply_channel_noise(test_cipher, noise_variance=(noise_val / 100.0))
            
            
        mode = self.cipher_mode.get()
        err_val = self.key_error_var.get()
        
        if mode == "DRPE":
            # 1. Apply Bandwidth Filter (Low-Pass)
            bandwidth = self.bandwidth_var.get()
            if bandwidth < 1.0:
                from filter_engine import apply_low_pass_filter
                filtered_channels = []
                
                # We have to filter all 3 color layers individually
                for i in range(3):
                    channel = test_cipher[:, :, i]
                    # Convert to waves, shift low frequencies to the center, filter, and convert back
                    freq = np.fft.fftshift(my_fft2(channel))
                    freq_filtered = apply_low_pass_filter(freq, bandwidth)
                    channel_filtered = my_ifft2(np.fft.ifftshift(freq_filtered))
                    filtered_channels.append(channel_filtered)
                    
                test_cipher = np.dstack(filtered_channels)

            # 2. Inject Key Error and Decrypt
            test_r2 = inject_key_error(self.r2, err_val) if err_val > 0 else self.r2
            self.decrypted_image = decrypt_image(test_cipher, self.r1, test_r2)
            
        elif mode == "ADVANCED":
            # Safety Check: Stop immediately if keys don't exist yet
            if self.advanced_keys is None:
                messagebox.showwarning("Warning", "No advanced keys found. Please encrypt an image in Advanced mode first.")
                return
                
            from advanced_drpe import decrypt_advanced
            
            # Unpack the 4 secret keys saved during encryption
            s1, s2, ax, ay = self.advanced_keys
            
            # Avalanche Attack: Add a microscopic error to the starting password
            if err_val > 0:
                s1 = s1 + (err_val * 0.00001)
                s2 = s2 + (err_val * 0.00001)  # Breaking the frequency mask destroys the structure
                
            # Get the bandwidth filter value from the slider
            bandwidth = self.bandwidth_var.get()
            
            # Decrypt using the new pipeline and extract the real physical magnitude!
            raw_decrypted = decrypt_advanced(test_cipher, s1, s2, ax, ay, bandwidth)
            self.decrypted_image = np.abs(raw_decrypted)



        # 5. Calculate MSE & Update UI
        if self.original_image is not None:
            mse = calculate_mse(self.original_image, self.decrypted_image)
            self.lbl_mse.config(text=f"Reconstruction MSE: {mse:.4e}")
        else:
            self.lbl_mse.config(text="Reconstruction MSE: N/A (Receiver Mode)")

        # Scale the decrypted output for the screen
        dec_display = self.decrypted_image / 255.0
        # Clip any math overflows so the image looks clean
        dec_display = np.clip(dec_display, 0, 1)

        self.ax3.clear()
        self.ax3.imshow(dec_display) 
        self.ax3.set_title("Decrypted Image", color="#FFFFFF") # Restored title
        self.ax3.axis("off") # Restored axis removal
        self.canvas.draw_idle()



    def export_keys(self):
        mode = self.cipher_mode.get()
        # Use different default filename based on mode
        default_filename = "seed.npz" if mode == "ADVANCED" else "key.npz"
        save_path = filedialog.asksaveasfilename(
            defaultextension=".npz",
            initialfile=default_filename,
            filetypes=[("NumPy Zip Archive", "*.npz")]
        )
        if not save_path:
            return

        if mode == "DRPE":
            if self.r1 is None or self.r2 is None:
                messagebox.showwarning("Warning", "No standard keys available to export.")
                return
            # Save the big 2D arrays
            np.savez(save_path, r1=self.r1, r2=self.r2)
            messagebox.showinfo("Export Successful", f"Keys saved to {os.path.basename(save_path)}")
            
        elif mode == "ADVANCED":
            if self.advanced_keys is None:
                messagebox.showwarning("Warning", "No advanced keys available to export.")
                return
            # Save the four dynamic decimal values
            s1, s2, ax, ay = self.advanced_keys
            np.savez(save_path, s1=s1, s2=s2, ax=ax, ay=ay)
            messagebox.showinfo("Export Successful", f"Seeds saved to {os.path.basename(save_path)}")


    def import_keys(self):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy Zip", "*.npz")])
        if not file_path:
            return
            
        data = np.load(file_path)
        
        # Check what kind of variables are saved inside the file
        if 's1' in data.files:
            # Rebuild the advanced_keys tuple from the saved decimals
            self.advanced_keys = (float(data['s1']), float(data['s2']), float(data['ax']), float(data['ay']))
            self.cipher_mode.set("ADVANCED")  # Auto-switch the UI
            messagebox.showinfo("Success", "Advanced Keys imported successfully.")
            
        elif 'r1' in data.files:
            # Load the big 2D standard keys
            self.r1, self.r2 = data['r1'], data['r2']
            self.cipher_mode.set("DRPE")  # Auto-switch the UI
            messagebox.showinfo("Success", "Standard Phase Keys imported successfully.")
            
        else:
            messagebox.showwarning("Error", "Unrecognized key format.")



    def export_ciphertext(self):
        if self.ciphertext is None:
            messagebox.showwarning("Warning", "No ciphertext to export.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".npy",
            initialfile="ciphertext.npy",
            filetypes=[("NumPy Array", "*.npy")]
        )
        if save_path:
            np.save(save_path, self.ciphertext)
            messagebox.showinfo("Success", "Ciphertext exported successfully.")



    def import_ciphertext(self):
        file_path = filedialog.askopenfilename(filetypes=[("NumPy Array", "*.npy")])
        if file_path:
            self.ciphertext = np.load(file_path)
            
            # Extract magnitude and scale it for the RGB screen
            cipher_display = np.abs(self.ciphertext)
            cipher_display = cipher_display / np.max(cipher_display)
            
            self.ax2.clear()
            self.ax2.imshow(cipher_display) # Removed cmap="inferno"
            self.ax2.set_title("Ciphertext |C(x,y)|", color="#FFFFFF")
            self.ax2.axis("off")
            self.canvas.draw_idle()
            self.lbl_mse.config(text="Status: Ciphertext Loaded.")

    def reset_all(self):
        """Clears all panels and resets state."""
        self.original_image = None
        self.ciphertext = None
        self.decrypted_image = None
        self.r1 = None
        self.r2 = None
        self.advanced_keys = None
        
        # Clear all axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        self.ax1.set_title("Original Image", color="#FFFFFF")
        self.ax2.set_title("Ciphertext (Magnitude)", color="#FFFFFF")
        self.ax3.set_title("Decrypted Output", color="#FFFFFF")
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.axis("off")
            ax.set_facecolor('#1E1E1E')
        
        self.canvas.draw_idle()
        
        self.lbl_global_status.config(text="Status: Reset. Awaiting image...", fg="#888888")
        self.lbl_mse.config(text="Reconstruction MSE: N/A")
        
    def exit_app(self):
        """Terminates the application."""
        self.root.quit()
        self.root.destroy()

    
if __name__ == "__main__":
    root = tk.Tk()
    app = DRPEApp(root)
    root.mainloop()