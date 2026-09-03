import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from drpe_cipher import generate_phase_mask, encrypt_image, decrypt_image, calculate_mse

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

        # Panel 1: Input & Encryption (Left)
        left_panel = tk.LabelFrame(
            main_frame,
            text=" Member 1: Input & Encryption Panel ",
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

        self.lbl_enc_status = tk.Label(
            left_panel, text="Status: Awaiting image...", font=("Consolas", 9),
            fg="#888888", bg="#1E1E1E"
        )
        self.lbl_enc_status.pack(pady=4)

        # Panel 2: Decryption & Testing (Right)
        right_panel = tk.LabelFrame(
            main_frame,
            text=" Member 2: Decryption & Testing Panel ",
            font=("Consolas", 11, "bold"),
            fg="#FF007F",
            bg="#1E1E1E",
            padx=10,
            pady=10
        )
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        btn_decrypt = tk.Button(
            right_panel, text="Decrypt Image", command=self.run_decryption,
            bg="#9B2226", fg="#FFFFFF", activebackground="#AE2012", activeforeground="#FFFFFF",
            font=("Consolas", 10, "bold"), width=18
        )
        btn_decrypt.pack(pady=4)

        btn_save_keys = tk.Button(
            right_panel, text="Export Keys (.npz)", command=self.export_keys,
            bg="#2A2A2A", fg="#FFFFFF", activebackground="#3A3A3A", activeforeground="#FFFFFF",
            font=("Consolas", 10), width=18
        )
        btn_save_keys.pack(pady=4)

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

        shape = self.original_image.shape
        self.r1 = generate_phase_mask(shape)
        self.r2 = generate_phase_mask(shape)
        self.lbl_enc_status.config(text="Status: Phase masks R1 & R2 generated.", fg="#00FF66")

    def run_encryption(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Load an image first.")
            return
        if self.r1 is None or self.r2 is None:
            self.generate_keys()

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

    def run_decryption(self):
        if self.ciphertext is None:
            messagebox.showwarning("Warning", "No ciphertext found. Run encryption first.")
            return

        self.decrypted_image = decrypt_image(self.ciphertext, self.r1, self.r2)

        # Calculate Mean Squared Error
        mse = calculate_mse(self.original_image, self.decrypted_image)
        self.lbl_mse.config(text=f"Reconstruction MSE: {mse:.4e}")

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

if __name__ == "__main__":
    root = tk.Tk()
    app = DRPEApp(root)
    root.mainloop()