from scipy.signal import decimate
import numpy as np
import matplotlib.pyplot as plt
import detect_pulse as pd

profiles = {
    "airspymini": {
        "filename": "../gnuradio/raw_iq_airspymini.bin",
        "sample_rate": 5e4 * 2
    },
    "airspyhf": {
        "filename": "../gnuradio/raw_iq_airspyhf_192000.bin",
        "sample_rate": 192e3
    }
}

selected_profile = "airspyhf"

# Load raw I/Q data (int16 interleaved)
data = np.fromfile(profiles[selected_profile]["filename"], dtype=np.complex64)
# data = np.fromfile("../Data/raw iq/raw_iq_airspyhf_192000.0.bin", dtype=np.complex64)

selected_pulse = 1
downsample_factor = 10
sample_rate = profiles[selected_profile]["sample_rate"]#/ downsample_factor

pulse_detector = pd.Pulse_Detector(samp_rate = sample_rate, verbose = True)

mags = np.abs(data)
mags_smooth = np.convolve(mags, np.ones(10)/10, mode='valid')


pulses = pulse_detector.detect([mags_smooth, data])

burst_start_sec = 2.33 *downsample_factor# seconds
burst_end_sec = 2.54  *downsample_factor # seconds
# burst_start_sec = 12.15 # seconds
# burst_end_sec = 12.34   # seconds
burst_start_sample = int(burst_start_sec * sample_rate)
burst_end_sample = int(burst_end_sec * sample_rate)

min_db = 0
max_db = -50
I = data.real
Q = data.imag

I_d = decimate(I, downsample_factor, ftype='fir', zero_phase=True)
Q_d = decimate(Q, downsample_factor, ftype='fir', zero_phase=True)

pulse_padding_s = 2e-4

pulse_start_secs = [] # seconds
pulse_end_secs = []   # seconds

for i in range(0, len(pulses)):
    pulse_start_time_s, pulse_dfreq_cubic, pulse_peak_db, pulse_noise_floor_db, pulse_snr_db, pulse_duration_ms, pulse_sample_lenth, pulse_buffer_size, pulse_dfreq_phasor, pulse_inter_ms = pulses[i].split(",")
    pulse_start_secs.append( float(pulse_start_time_s) - pulse_padding_s )
    pulse_end_secs.append( float(pulse_start_time_s) + (float(pulse_duration_ms) / 1e3) + pulse_padding_s )
    if (max_db < float(pulse_peak_db)):
        max_db = float(pulse_peak_db)
    if (min_db > float(pulse_noise_floor_db)):
        min_db = float(pulse_noise_floor_db)

n_plots = len(pulse_start_secs) 

print("Number of plots: ", n_plots)

if n_plots > 0:
    fig, axs = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)
    for i in range(0, n_plots):
        plt.subplot(n_plots, 1, i + 1)

        ax = axs[i]

        pulse_start_sec = pulse_start_secs[i] # seconds
        pulse_end_sec = pulse_end_secs[i] # seconds
        pulse_start_sample = int(pulse_start_sec * sample_rate)
        pulse_end_sample = int(pulse_end_sec * sample_rate)
        pulse_duration_samples = pulse_end_sample - pulse_start_sample

        print(f"Plotting pulse starting at {pulse_start_sec} s and ending at {pulse_end_sec} s.")
        pulse_duration_ms = (pulse_end_sec - pulse_start_sec) * 1e3

        pulse_mags = mags_smooth[pulse_start_sample:pulse_end_sample]
        pulse_slope = np.diff(pulse_mags) * 25

        pulse_padding_samples = int(pulse_padding_s * sample_rate)
        # Plot time-domain I/Q
        ax.plot(I[pulse_start_sample:pulse_end_sample], label='I')
        ax.plot(Q[pulse_start_sample:pulse_end_sample], label='Q')
        ax.plot(pulse_mags, label='Magnitude')
        ax.plot(pulse_slope, label='Magnitude')
        ax.axvline( x = pulse_padding_samples, color = 'red' )
        ax.axvline( x = pulse_duration_samples - pulse_padding_samples, color = 'blue' )


        # plt.plot(I[burst_start_sample:burst_end_sample], label='I')
        # plt.plot(Q[burst_start_sample:burst_end_sample], label='Q')
        ax.set_title(f"Time-domain Magnitude ({pulse_duration_ms - pulse_padding_s*2*1000} ms)")
        ax.grid(True)

""" start = int(0.86 * 3e6)
end = int(0.87 * 3e6)
segment = data[start:end] """

# Plot spectrogram
plt.figure(figsize=(10, 4))
plt.specgram((data), NFFT=1024, Fs=sample_rate, noverlap=1, scale='dB', vmin=min_db - 100, vmax=max_db - 60)
# plt.xlim(burst_start_sec,burst_end_sec)
plt.ylim(-8e3,8e3)
plt.title(f"Spectrogram for {len(data)} samples")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(label='dB')



plt.tight_layout()
plt.show()