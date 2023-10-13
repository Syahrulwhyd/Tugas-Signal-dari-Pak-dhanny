# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:37:59 2023

@author: InBook X1
"""

import numpy as np
import matplotlib.pyplot as plt

# Fungsi FFT 1D (Dekomposisi Koefisien)
def fft1D(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft1D(x[0::2])
    odd = fft1D(x[1::2])
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# Contoh data dalam 1D
N = 64
t = np.linspace(0.0, 2.0 * np.pi, N)
f = 5.0  # Frekuensi sinyal
signal = np.sin(f * t) + 0.5 * np.sin(2.0 * f * t)  # Gabungan dua gelombang sinus

# Terapkan FFT 1D menggunakan NumPy
fft_result_numpy = np.fft.fft(signal)

# Terapkan FFT 1D secara manual
fft_result_manual = fft1D(signal)

# Plot sinyal asli
plt.figure(figsize=(12, 6))
plt.subplot(131)
plt.title("Sinyal Asli")
plt.plot(t, signal)

# Plot magnitude dari hasil FFT dengan NumPy
magnitude_numpy = np.abs(fft_result_numpy)
plt.subplot(132)
plt.title("Magnitude FFT (NumPy)")
plt.plot(magnitude_numpy)

# Plot magnitude dari hasil FFT secara manual
magnitude_manual = np.abs(fft_result_manual)
plt.subplot(133)
plt.title("Magnitude FFT (Manual)")
plt.plot(magnitude_manual)

plt.tight_layout()
plt.show()

# Membandingkan hasil dengan NumPy
if np.allclose(fft_result_manual, fft_result_numpy):
    print("\nHasil FFT manual sesuai dengan hasil NumPy.")
else:
    print("\nHasil FFT manual tidak sesuai dengan hasil NumPy.")

