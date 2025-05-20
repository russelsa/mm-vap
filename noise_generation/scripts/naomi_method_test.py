import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Parameters
tgt_snr = -5  # in dB
rms_thresh = 0.05  # use top fraction of RMS values measured in 1 sec chunks

# Load audio files
x, fs = sf.read('data/speech_left.wav')
x_no_sil, fs2 = sf.read('data/speech_no_silences_left_channel.wav')
nse, fs_nse = sf.read('data/noise.wav')

# Compute RMS values
rms_speech = np.sqrt(np.mean(x_no_sil**2))
rms_nse = np.sqrt(np.mean(nse**2))

# Calculate steady-state alpha
alpha_steady = rms_speech / (rms_nse * (10**(tgt_snr / 20)))
x_noisy_naomi = x + (alpha_steady * nse)

# Compute RMS in 1-second chunks
num_secs = len(x_no_sil) // fs
track_rms = []

for i in range(num_secs - 1):
    data = x_no_sil[i * fs: (i + 1) * fs]
    data_rms = np.sqrt(np.mean(data**2))
    track_rms.append(data_rms)

# # Plot the RMS tracking
# plt.plot(track_rms)
# plt.title('RMS Tracking')
# plt.xlabel('Time (s)')
# plt.ylabel('RMS')
# plt.savefig("rms.png")

# Order RMS values starting with the highest
track_rms = sorted(track_rms, reverse=True)
cutoff = int(num_secs * rms_thresh)
use_rms = track_rms[:cutoff]

# Compute target alpha for higher power elements
alpha = np.mean(use_rms) / (rms_nse * (10**(tgt_snr / 20)))
x_noisy_naomi = x + (alpha * nse)

# Normalize the noisy signal
x_noisy_naomi_norm = x_noisy_naomi / np.max(np.abs(x_noisy_naomi))

# Save the noisy audio
sf.write('data/noisy_0db_naomi_top_rms.wav', x_noisy_naomi_norm, fs)
