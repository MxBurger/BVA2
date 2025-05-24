import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from Samples import create_synthetic_image

def add_noise(image, noise_variance):
    """Fügt Gaußsches Rauschen hinzu"""
    noise = np.random.normal(0, np.sqrt(noise_variance), image.shape)
    return np.clip(image + noise, 0, 255).astype(np.uint8)

def degrade_image(image, kernel, noise_variance):
    """Degradiert Bild durch Blur + Rauschen"""
    blurred = cv2.filter2D(image.astype(np.float64), -1, kernel)
    return add_noise(blurred, noise_variance)

def calculate_psnr(original, restored):
    """Berechnet PSNR"""
    mse = np.mean((original.astype(float) - restored.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255 ** 2 / mse)

# Lade oder erstelle Testbild
try:
    img = cv2.imread("boat.tif", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError
except:
    img = create_synthetic_image()

# Erstelle Blur-Kernel (Gaussian)
kernel_size = 15
sigma = kernel_size / 3.0
ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
xx, yy = np.meshgrid(ax, ax)
kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
kernel /= kernel.sum()

# Degradiere Bild
noise_var = 50
degraded = degrade_image(img, kernel, noise_var)

# Wende SciPy Wiener Filter an
filtered_signal = signal.wiener(degraded, (5, 5))
filtered_signal = np.clip(filtered_signal, 0, 255).astype(np.uint8)

# Berechne PSNR-Werte
original_psnr = float('inf')
degraded_psnr = calculate_psnr(img, degraded)
filtered_psnr = calculate_psnr(img, filtered_signal)

# Zeige Ergebnisse
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(degraded, cmap='gray')
axes[1].set_title(f'Degradiert\nPSNR: {degraded_psnr:.1f} dB')
axes[1].axis('off')

axes[2].imshow(filtered_signal, cmap='gray')
axes[2].set_title(f'Wiener gefiltert\nPSNR: {filtered_psnr:.1f} dB')
axes[2].axis('off')

plt.tight_layout()
plt.show()

print(f"PSNR Vergleich:")
print(f"Degradiert: {degraded_psnr:.2f} dB")
print(f"Nach Wiener Filter: {filtered_psnr:.2f} dB")
print(f"Verbesserung: {filtered_psnr - degraded_psnr:.2f} dB")