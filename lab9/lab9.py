import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, get_window
import soundfile as sf
import pandas as pd
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "sound.wav"
OUTPUT_DIR = BASE_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_DURATION = 0.05
OVERLAP = 0.75
DT = 0.1
DF = 50
MIN_FREQ = 20
MAX_FREQ = 5000


signal, sr = sf.read(str(INPUT_FILE))

if signal.ndim > 1:
    signal = signal.mean(axis=1)

N = len(signal)
duration = N / sr


nperseg = int(WINDOW_DURATION * sr)
noverlap = int(nperseg * OVERLAP)
window = get_window('hann', nperseg)

f, t, Zxx = stft(signal, fs=sr, window=window,
                 nperseg=nperseg, noverlap=noverlap)

magnitude = np.abs(Zxx)


freq_mask = (f >= MIN_FREQ) & (f <= MAX_FREQ)
f_cut = f[freq_mask]
magnitude_cut = magnitude[freq_mask, :]


plt.figure(figsize=(10, 5))
plt.pcolormesh(t, f_cut, 20*np.log10(magnitude_cut + 1e-10),
               shading='gouraud', cmap='inferno')
plt.title("Spectrogram before")
plt.xlabel("Time, s")
plt.ylabel("Frequency, Hz")
plt.yscale("log")
plt.ylim(MIN_FREQ, MAX_FREQ)
plt.colorbar(label="dB")
plt.savefig(OUTPUT_DIR / "spectrogram_before.png")
plt.close()


energy = np.sum(magnitude_cut**2, axis=0)
threshold = np.percentile(energy, 20)
noise_frames = magnitude_cut[:, energy < threshold]

noise_spectrum = np.mean(noise_frames, axis=1, keepdims=True)


k = 0.8

clean_magnitude_cut = magnitude_cut - k * noise_spectrum
clean_magnitude_cut = np.maximum(clean_magnitude_cut, 0.05 * magnitude_cut)

Zxx_clean = np.zeros_like(Zxx, dtype=complex)
Zxx_clean[freq_mask, :] = clean_magnitude_cut * np.exp(1j * np.angle(Zxx[freq_mask, :]))


_, signal_clean = istft(Zxx_clean, fs=sr,
                        window=window,
                        nperseg=nperseg,
                        noverlap=noverlap)

signal_clean = signal_clean[:len(signal)]


sf.write(str(OUTPUT_DIR / "denoised.wav"), signal_clean, sr)


f2, t2, Zxx2 = stft(signal_clean, fs=sr,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap)

magnitude2 = np.abs(Zxx2)[freq_mask, :]

plt.figure(figsize=(10, 5))
plt.pcolormesh(t2, f_cut, 20*np.log10(magnitude2 + 1e-10),
               shading='gouraud', cmap='inferno')
plt.title("Spectrogram after")
plt.xlabel("Time, s")
plt.ylabel("Frequency, Hz")
plt.yscale("log")
plt.ylim(MIN_FREQ, MAX_FREQ)
plt.colorbar(label="dB")
plt.savefig(OUTPUT_DIR / "spectrogram_after.png")
plt.close()


plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(signal)) / sr, signal)
plt.title("Waveform before")
plt.xlabel("Time, s")
plt.ylabel("Amplitude")
plt.savefig(OUTPUT_DIR / "waveform.png")
plt.close()

plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(signal_clean)) / sr, signal_clean)
plt.title("Waveform after")
plt.xlabel("Time, s")
plt.ylabel("Amplitude")
plt.savefig(OUTPUT_DIR / "waveform_denoised.png")
plt.close()

noise_est = signal - signal_clean

snr_before = 10 * np.log10(np.sum(signal**2) / np.sum(noise_est**2))
snr_after = 10 * np.log10(np.sum(signal_clean**2) / np.sum(noise_est**2))


power = magnitude_cut**2
time_bins = np.arange(0, duration, DT)

peaks = []

for i in range(len(time_bins) - 1):
    t_mask = (t >= time_bins[i]) & (t < time_bins[i + 1])

    if not np.any(t_mask):
        continue

    block = power[:, t_mask]
    avg_spectrum = np.mean(block, axis=1)

    j = np.argmax(avg_spectrum)

    peaks.append([
        time_bins[i],
        time_bins[i + 1],
        f_cut[j],
        f_cut[j] + DF,
        avg_spectrum[j]
    ])

peaks = sorted(peaks, key=lambda x: x[4], reverse=True)


plt.figure(figsize=(10, 5))
plt.pcolormesh(t, f_cut, 20*np.log10(magnitude_cut + 1e-10),
               shading='gouraud', cmap='inferno')

top_peaks = peaks[:5]

for p in top_peaks:
    t_center = (p[0] + p[1]) / 2
    f_center = (p[2] + p[3]) / 2
    plt.scatter(t_center, f_center, color='cyan', s=70)

plt.title("Spectrogram with peaks")
plt.xlabel("Time, s")
plt.ylabel("Frequency, Hz")
plt.yscale("log")
plt.ylim(MIN_FREQ, MAX_FREQ)
plt.colorbar(label="dB")
plt.savefig(OUTPUT_DIR / "spectrogram_with_peaks.png")
plt.close()


df = pd.DataFrame(peaks, columns=["t1", "t2", "f1", "f2", "E"])
df.to_csv(OUTPUT_DIR / "energy_peaks.csv", index=False)


print("Sample rate:", sr)
print("Duration:", duration)
print("SNR before:", snr_before)
print("SNR after:", snr_after)
print("Top-5 peaks:")
for p in top_peaks:
    print(p)
