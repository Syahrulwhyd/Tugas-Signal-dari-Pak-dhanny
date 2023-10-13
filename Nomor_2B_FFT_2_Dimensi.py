# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:23:43 2023

@author: InBook X1
"""

import numpy as np
import matplotlib.pyplot as plt

# Membuat contoh citra 2D
x = np.linspace(0, 2 * np.pi, 256)
y = np.linspace(0, 2 * np.pi, 256)
X, Y = np.meshgrid(x, y)
image = np.sin(X) + np.cos(Y)

# Terapkan FFT 2D menggunakan NumPy
fft_result = np.fft.fft2(image)

# Hitung spektrum frekuensi
magnitude = np.abs(fft_result)

# Plot gambar asli
plt.figure(figsize=(10, 4))
plt.subplot(131)
plt.title("Gambar Asli")
plt.imshow(image, cmap='gray')

# Plot spektrum frekuensi
plt.subplot(132)
plt.title("Spektrum Frekuensi")
plt.imshow(np.fft.fftshift(magnitude), cmap='viridis')

# Plot fase
plt.subplot(133)
plt.title("Fase")
plt.imshow(np.angle(fft_result), cmap='hsv')

plt.tight_layout()
plt.show()


