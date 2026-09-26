import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import font as tkfont
import numpy as np
from PIL import Image
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from drpe_cipher import generate_phase_mask, encrypt_image, decrypt_image,calculate_mse , inject_key_error, apply_cropping_attack, apply_channel_noise
from fourier_transforms import my_fft2, my_ifft2


# =============================================================================
# THEME  (single source of truth for every colour used in the UI and the plots)
# =============================================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

BG        = "#0B0E14"   # window background
CARD      = "#141922"   # panels + plot card (the Matplotlib figure uses this too)
CARD_EDGE = "#232B3A"   # 1px card outline
INSET     = "#0E1218"   # recessed pills (status / MSE read-outs)
TRACK     = "#2A3244"   # slider tracks, radio outlines
TEXT      = "#E6EAF2"
TEXT_DIM  = "#8A93A6"

GREEN = "#00FF66"       # brand / success  (your original header + status colour)
CYAN  = "#00E5FF"       # encryption panel
PINK  = "#FF007F"       # decryption panel
AMBER = "#FFB703"       # MSE read-out

# (normal, hover) pairs - taken from your original button colours
BTN_NEUTRAL = ("#232B3A", "#313C52")
BTN_PRIMARY = ("#005F73", "#0A9396")
BTN_DANGER  = ("#9B2226", "#AE2012")
BTN_RESET   = ("#FF6B35", "#FF8C5A")
BTN_EXIT    = ("#DC2F02", "#F77F00")

RADIUS = 12             # corner radius for every button


# -----------------------------------------------------------------------------
# Compatibility shims.
# Your existing methods call  widget.config(text=..., fg=...)  (the classic Tk
# API).  CustomTkinter widgets use .configure(text=..., text_color=...), so these
# two thin subclasses translate the old option names.  This is what lets every
# one of your functions stay exactly as you wrote it.
# -----------------------------------------------------------------------------
class _TkCompat:
    _ALIASES = {"fg": "text_color", "bg": "fg_color"}

    def configure(self, require_redraw=False, **kwargs):
        for old, new in self._ALIASES.items():
            if old in kwargs:
                kwargs[new] = kwargs.pop(old)
        super().configure(require_redraw=require_redraw, **kwargs)

    config = configure


class UILabel(_TkCompat, ctk.CTkLabel):
    pass


class UIButton(_TkCompat, ctk.CTkButton):
    pass


class DRPEApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Fourier-Domain Image Encryptor (DRPE)")
        self.root.geometry("1240x820")
        self.root.minsize(1100, 740)
        self.root.configure(fg_color=BG)

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

    # ------------------------------------------------------------------
    # UI-only helpers (no logic - they just build widgets)
    # ------------------------------------------------------------------
    def _font(self, size, weight="normal", mono=False):
        family = self._mono_family if mono else self._ui_family
        return ctk.CTkFont(family=family, size=size, weight=weight)

    def _ui_button(self, parent, text, command, colors, bold=False, height=42):
        return UIButton(
            parent, text=text, command=command,
            height=height, corner_radius=RADIUS,
            fg_color=colors[0], hover_color=colors[1], text_color="#FFFFFF",
            font=self._font(13, "bold" if bold else "normal"),
        )

    def _panel(self, parent, column):
        panel = ctk.CTkFrame(
            parent, fg_color=CARD, corner_radius=16,
            border_width=1, border_color=CARD_EDGE,
        )
        panel.grid(row=0, column=column, sticky="nsew", padx=6)
        panel.grid_columnconfigure((0, 1), weight=1, uniform=f"panel{column}")
        return panel

    def _panel_header(self, panel, text, color):
        row = ctk.CTkFrame(panel, fg_color="transparent")
        row.grid(row=0, column=0, columnspan=2, sticky="ew", padx=18, pady=(16, 8))
        ctk.CTkFrame(row, width=4, height=18, corner_radius=2, fg_color=color).pack(side="left")
        ctk.CTkLabel(
            row, text=text, text_color=color, font=self._font(15, "bold"),
        ).pack(side="left", padx=(10, 0))

    def _pill(self, parent, row, pady):
        pill = ctk.CTkFrame(parent, fg_color=INSET, corner_radius=10)
        pill.grid(row=row, column=0, columnspan=2, sticky="ew", padx=16, pady=pady)
        return pill

    def _build_slider(self, parent, row, column, label_text, variable,
                      from_, to, steps, accent, fmt, label_color=TEXT):
        """Label on top, slider + live value read-out underneath."""
        cell = ctk.CTkFrame(parent, fg_color="transparent")
        left = 16 if column == 0 else 8
        right = 8 if column == 0 else 16
        cell.grid(row=row, column=column, sticky="ew", padx=(left, right), pady=5)
        cell.grid_columnconfigure(0, weight=1)

        label = UILabel(cell, text=label_text, text_color=label_color,
                        font=self._font(12), anchor="w")
        label.grid(row=0, column=0, columnspan=2, sticky="w")

        slider = ctk.CTkSlider(
            cell, variable=variable, from_=from_, to=to, number_of_steps=steps,
            height=16, fg_color=TRACK, progress_color=accent,
            button_color=TEXT, button_hover_color="#FFFFFF",
        )
        slider.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        readout = ctk.CTkLabel(
            cell, text=format(variable.get(), fmt), width=46, anchor="e",
            font=self._font(12, "bold", mono=True), text_color=accent,
        )
        readout.grid(row=1, column=1, padx=(8, 0), pady=(5, 0))
        variable.trace_add("write", lambda *_: readout.configure(text=format(variable.get(), fmt)))
        return label, slider, readout

    # ------------------------------------------------------------------
    # MAIN LAYOUT
    # ------------------------------------------------------------------
    def _setup_ui(self):
        # Pick the best fonts the current OS actually has
        installed = set(tkfont.families(self.root))
        self._ui_family = next((f for f in ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter", "Roboto", "DejaVu Sans")
                                if f in installed), "Arial")
        self._mono_family = next((f for f in ("Consolas", "Menlo", "DejaVu Sans Mono", "Courier New")
                                  if f in installed), "Courier")

        # Root grid: header / mode bar / control panels / plots (plots take the spare space)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(3, weight=1)

        # ---------------- Header: title (left) + Reset / Exit (right) ----------------
        title_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 6))
        title_frame.grid_columnconfigure(0, weight=1)

        header = UILabel(
            title_frame,
            text="FOURIER-DOMAIN IMAGE ENCRYPTOR (DRPE)",
            font=self._font(24, "bold", mono=True),
            text_color=GREEN,
        )
        header.grid(row=0, column=0, sticky="w")

        button_frame = ctk.CTkFrame(title_frame, fg_color="transparent")
        button_frame.grid(row=0, column=1, sticky="e")

        btn_reset = UIButton(
            button_frame, text="Reset", command=self.reset_all,
            width=92, height=36, corner_radius=18,
            fg_color=BTN_RESET[0], hover_color=BTN_RESET[1], text_color="#FFFFFF",
            font=self._font(13, "bold"),
        )
        btn_reset.pack(side="left", padx=4)

        btn_exit = UIButton(
            button_frame, text="Exit", command=self.exit_app,
            width=92, height=36, corner_radius=18,
            fg_color=BTN_EXIT[0], hover_color=BTN_EXIT[1], text_color="#FFFFFF",
            font=self._font(13, "bold"),
        )
        btn_exit.pack(side="left", padx=4)

        # ---------------- Algorithm selector mode ----------------
        #added after nprs
        self.cipher_mode = tk.StringVar(value="DRPE")
        self.cipher_mode.trace_add("write", lambda *args: self._update_key_button_labels())

        mode_frame = ctk.CTkFrame(
            self.root, fg_color=CARD, corner_radius=14,
            border_width=1, border_color=CARD_EDGE,
        )
        mode_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=6)

        ctk.CTkLabel(
            mode_frame, text="Cipher mode", text_color=TEXT_DIM, font=self._font(13),
        ).pack(side="left", padx=(20, 24), pady=12)

        ctk.CTkRadioButton(
            mode_frame, text="Standard DRPE (Linear)", variable=self.cipher_mode, value="DRPE",
            fg_color="#00C853", hover_color="#00E676", border_color=TRACK,
            text_color="#3DDC84", font=self._font(13, "bold"),
        ).pack(side="left", padx=(0, 28), pady=12)

        ctk.CTkRadioButton(
            mode_frame, text="Advanced DRPE", variable=self.cipher_mode, value="ADVANCED",
            fg_color=PINK, hover_color="#FF3D9A", border_color=TRACK,
            text_color="#FF4DA6", font=self._font(13, "bold"),
        ).pack(side="left", pady=12)

        # ---------------- Control panels (side by side) ----------------
        main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        main_frame.grid(row=2, column=0, sticky="ew", padx=14, pady=4)
        main_frame.grid_columnconfigure((0, 1), weight=1, uniform="panels")
        main_frame.grid_rowconfigure(0, weight=1)

        # Panel 1: Input & Encryption (Left)
        left_panel = self._panel(main_frame, column=0)
        self._panel_header(left_panel, "Input & Encryption Panel", CYAN)

        btn_load = self._ui_button(left_panel, "Load Image", self.load_image, BTN_NEUTRAL)
        btn_load.grid(row=1, column=0, sticky="ew", padx=(16, 6), pady=6)

        btn_gen_keys = self._ui_button(left_panel, "Generate Phase Keys", self.generate_keys, BTN_NEUTRAL)
        btn_gen_keys.grid(row=1, column=1, sticky="ew", padx=(6, 16), pady=6)

        btn_encrypt = self._ui_button(left_panel, "Encrypt Image", self.run_encryption, BTN_PRIMARY, bold=True)
        btn_encrypt.grid(row=2, column=0, sticky="ew", padx=(16, 6), pady=6)

        # new button's added here to export ciphertext
        #
        #
        btn_export_ciphertext = self._ui_button(left_panel, "Export CipherText", self.export_ciphertext, BTN_PRIMARY, bold=True)
        btn_export_ciphertext.grid(row=2, column=1, sticky="ew", padx=(6, 16), pady=6)

        btn_export_keys = self._ui_button(left_panel, "Export Keys (.npz)", self.export_keys, BTN_NEUTRAL)
        btn_export_keys.grid(row=3, column=0, columnspan=2, sticky="ew", padx=16, pady=6)
        self.btn_export_keys = btn_export_keys  # Keep reference to update label

        # Status message pill - fixed slot so the layout never jumps
        status_frame = self._pill(left_panel, row=4, pady=(10, 16))
        self.lbl_global_status = UILabel(
            status_frame, text="Status: Awaiting image...",
            font=self._font(12, mono=True), text_color="#888888",
            anchor="center", justify="center", wraplength=480,
        )
        self.lbl_global_status.pack(fill="both", expand=True, padx=14, pady=9)

        left_panel.grid_rowconfigure((1, 2, 3), weight=1)

        # Panel 2: Decryption & Testing (Right)
        right_panel = self._panel(main_frame, column=1)
        self._panel_header(right_panel, "Decryption & Testing Panel", PINK)

        #buttons from importing keys and cyphertext
        btn_import_ciphertext = self._ui_button(right_panel, "Import CipherText", self.import_ciphertext, BTN_PRIMARY, bold=True)
        btn_import_ciphertext.grid(row=1, column=0, sticky="ew", padx=(16, 6), pady=6)

        btn_import_keys = self._ui_button(right_panel, "Import Keys (.npz)", self.import_keys, BTN_NEUTRAL)
        btn_import_keys.grid(row=1, column=1, sticky="ew", padx=(6, 16), pady=6)
        self.btn_import_keys = btn_import_keys  # Keep reference to update label

        btn_decrypt = self._ui_button(right_panel, "Decrypt Image", self.run_decryption, BTN_DANGER, bold=True)
        btn_decrypt.grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=6)

        ctk.CTkFrame(right_panel, height=1, corner_radius=0, fg_color=CARD_EDGE).grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=16, pady=(10, 6))

        # --- Sayem CONTROLS ---
        # Key Sensitivity Slider
        self.lbl_key_error, self.scale_error, _ = self._build_slider(
            right_panel, row=4, column=0, label_text="Key Error %:",
            variable=self.key_error_var, from_=0.0, to=5.0, steps=50,
            accent=PINK, fmt=".1f",
        )

        # --- ATTACK SLIDERS ---
        # Cropping Attack Slider (0% to 100%)
        self.crop_val_var = tk.DoubleVar(value=0.0)
        _, self.scale_crop, _ = self._build_slider(
            right_panel, row=4, column=1, label_text="Crop Area %:",
            variable=self.crop_val_var, from_=0.0, to=100.0, steps=100,
            accent=PINK, fmt=".0f",
        )

        # Noise Attack Slider (0% to 100%)
        self.noise_val_var = tk.DoubleVar(value=0.0)
        _, self.scale_noise, _ = self._build_slider(
            right_panel, row=5, column=0, label_text="Noise Power %:",
            variable=self.noise_val_var, from_=0.0, to=100.0, steps=100,
            accent=PINK, fmt=".0f",
        )
        # -----------------------------

        # Bandwidth / Low-Pass Filter Slider
        self.bandwidth_var = tk.DoubleVar(value=1.0)
        self._build_slider(
            right_panel, row=5, column=1, label_text="Bandwidth % (1.0 = Perfect):",
            variable=self.bandwidth_var, from_=0.01, to=1.0, steps=99,
            accent=GREEN, fmt=".2f", label_color=GREEN,
        )

        mse_frame = self._pill(right_panel, row=6, pady=(10, 16))
        self.lbl_mse = UILabel(
            mse_frame, text="Reconstruction MSE: N/A", font=self._font(13, "bold", mono=True),
            text_color=AMBER, anchor="w", justify="left",
        )
        self.lbl_mse.pack(fill="x", padx=14, pady=9)

        # ---------------- Matplotlib display area (bottom) ----------------
        # Dark theme applied globally so that ax.clear() in your functions keeps it.
        plt.rcParams.update({
            "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD,
            "axes.edgecolor": CARD, "axes.linewidth": 0,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.spines.bottom": False, "axes.spines.left": False,
            "text.color": TEXT, "axes.labelcolor": TEXT,
            "xtick.color": TEXT_DIM, "ytick.color": TEXT_DIM,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlepad": 10,
        })

        plot_card = ctk.CTkFrame(
            self.root, fg_color=CARD, corner_radius=16,
            border_width=1, border_color=CARD_EDGE,
        )
        plot_card.grid(row=3, column=0, sticky="nsew", padx=20, pady=(6, 18))

        # Figure (not pyplot.subplots) so Matplotlib doesn't open a hidden second Tk window
        self.fig = Figure(figsize=(14, 4))
        self.ax1, self.ax2, self.ax3 = self.fig.subplots(1, 3)
        self.fig.patch.set_facecolor(CARD)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.06)

        for ax, title in zip([self.ax1, self.ax2, self.ax3], ["Original Image", "Ciphertext (Magnitude)", "Decrypted Output"]):
            ax.set_title(title, color="#FFFFFF")
            ax.axis("off")
            ax.set_facecolor(CARD)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_card)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.configure(bg=CARD, highlightthickness=0, bd=0)
        canvas_widget.pack(fill="both", expand=True, padx=12, pady=12)

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
    root = ctk.CTk()
    app = DRPEApp(root)
    root.mainloop()