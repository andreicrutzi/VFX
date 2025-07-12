import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import io
import base64

class ImageEffectsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Efecte speciale pe imagini color")
        self.image = None
        self.processed_image = None

        self.label = tk.Label(root, text="Selectează o imagine și un efect")
        self.label.pack(pady=10)

        self.btn_load = tk.Button(root, text="Încarcă imagine", command=self.load_image)
        self.btn_load.pack(pady=5)

        self.effect_var = tk.StringVar()
        self.effects = ["Sepia Filter", "Gamma Correction (0.5)", "Vignette Effect", "Anaglyph", "Duo-Tone", "Mosaic Effect"]
        self.effect_combobox = ttk.Combobox(root, textvariable=self.effect_var, values=self.effects, state="readonly")
        self.effect_combobox.pack(pady=5)
        self.effect_combobox.set("Alege un efect")

        self.btn_apply = tk.Button(root, text="Aplică efect", command=self.apply_effect)
        self.btn_apply.pack(pady=5)

        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.display_image(self.image)

    def display_image(self, img):
        height, width = img.shape[:2]
        ratio = min(400 / width, 400 / height)
        new_size = (int(width * ratio), int(height * ratio))
        img_resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        img_pil = Image.fromarray(img_resized)
        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(200, 200, image=self.photo, anchor="center")

    def apply_effect(self):
        if self.image is None:
            tk.messagebox.showerror("Eroare", "Încarcă o imagine mai întâi!")
            return

        effect = self.effect_var.get()
        if effect == "Alege un efect":
            tk.messagebox.showerror("Eroare", "Selectează un efect!")
            return

        self.processed_image = self.image.copy()

        if effect == "Sepia Filter":
            self.processed_image = self.sepia_filter(self.processed_image)
        elif effect == "Gamma Correction (gamma = 0.5)":
            self.processed_image = self.gamma_correction(self.processed_image, gamma=0.5)
        elif effect == "Vignette Effect":
            self.processed_image = self.vignette_effect(self.processed_image)
        elif effect == "Anaglyph":
            self.processed_image = self.anaglyph_effect(self.processed_image)
        elif effect == "Duo-Tone":
            self.processed_image = self.duo_tone_effect(self.processed_image)
        elif effect == "Mosaic Effect":
            self.processed_image = self.mosaic_effect(self.processed_image)

        self.display_image(self.processed_image)

    def sepia_filter(self, img):
        sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        sepia_img = cv2.transform(img, sepia_matrix)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return sepia_img

    def gamma_correction(self, img, gamma):
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)

    def vignette_effect(self, img):
        rows, cols = img.shape[:2]
        X_resultant = np.square(np.array(range(cols)) - cols / 2)
        Y_resultant = np.square(np.array(range(rows)) - rows / 2)
        X_resultant, Y_resultant = np.meshgrid(X_resultant, Y_resultant)
        resultant = X_resultant + Y_resultant
        resultant = resultant / np.max(resultant)
        factor = 1 - resultant
        vignette = np.zeros_like(img, dtype=np.float32)
        for i in range(3):
            vignette[:, :, i] = img[:, :, i] * factor
        return np.clip(vignette, 0, 255).astype(np.uint8)

    def anaglyph_effect(self, img):
        rows, cols = img.shape[:2]
        left_img = np.roll(img, -10, axis=1)
        right_img = np.roll(img, 10, axis=1)
        anaglyph = np.zeros_like(img)
        anaglyph[:, :, 0] = left_img[:, :, 0]
        anaglyph[:, :, 1] = right_img[:, :, 1]
        anaglyph[:, :, 2] = right_img[:, :, 2]
        return anaglyph

    def duo_tone_effect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        duo_tone = np.zeros_like(img)
        color1 = np.array([255, 0, 0])  # Red
        color2 = np.array([0, 0, 255])  # Blue
        mask = gray < 128
        duo_tone[mask] = color1
        duo_tone[~mask] = color2
        return duo_tone

    def mosaic_effect(self, img, block_size=10):
        #nimic ca nu am apucat
        return img

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEffectsApp(root)
    root.mainloop()