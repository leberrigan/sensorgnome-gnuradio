from scipy.signal import decimate
import numpy as np
import matplotlib.pyplot as plt
import detect_pulse as pd

selected_profile = "funcube"

profiles = {
    "airspymini": {
        "filename": "raw_iq_airspymini_50000.bin",
        "sample_rate": 50000
    },
    "airspyhf": {
        "filename": "raw_iq_airspyhf_192000.bin",
        "sample_rate": 192000
    },
    "rtlsdr": {
        "filename": "raw_iq_rtlsdr_120000.bin",
        "sample_rate": 120000
    },
    "funcube": {
        "filename": "raw_iq_funcube_48000.bin",
        "sample_rate": 48000
    }
}

class Pulse_Analysis:
    def __init__(self):
                
        self.sample_rate = profiles[selected_profile]["sample_rate"]#/ downsample_factor
        # Load raw I/Q data (int16 interleaved)
        self.raw_iq = np.fromfile(profiles[selected_profile]["filename"], dtype=np.complex64)
        # data = np.fromfile("../Data/raw iq/raw_iq_airspyhf_192000.0.bin", dtype=np.complex64)

        pulse_detector = pd.Pulse_Detector(samp_rate = self.sample_rate, verbose = True, min_snr_db=5, debounce_samples=10, pulse_len_ms=2.5)

        mags = np.abs(self.raw_iq)
        # Get moving average of magnitude
        self.mags_smooth = np.convolve(mags, np.ones(3)/3, mode='valid')
        
        self.pulses = pulse_detector.detect([self.mags_smooth, self.raw_iq])

        if len(self.pulses) == 0:
            print("No pulses detected.")
            return
            
        self.pulse_padding_s = 2e-4
        self.min_fft_size = 64
        self.min_db = 0
        self.max_db = -50

        pulse_times = self.get_pulse_times()

        n_pulses = len(self.pulses)
        n_plots = 2
        figure, self.axes = plt.subplots(n_pulses, n_plots, figsize=(5 * n_plots, 3 * n_pulses), sharex='col')

        print("Number of pulses:", n_pulses)
        for i in range(n_pulses):
            for j in range(n_plots):
                plot_idx = i * 2 + j
                if (j == 0):
                    # plt.subplot(n_pulses, 2, plot_idx + 1)
                    ax = self.axes[i,j]
                    self.plot_pulse_iq(pulse_times[i], ax)
                elif (j == 1):
                    # plt.subplot(n_pulses, 2, plot_idx + 1)
                    ax = self.axes[i,j]
                    self.plot_pulse_fft(pulse_times[i], ax)

        plt.tight_layout()
        plt.show()

    def downsample(self, downsample_factor = 10):
        I_d = decimate(self.raw_iq.real, downsample_factor, ftype='fir', zero_phase=True)
        Q_d = decimate(self.raw_iq.imag, downsample_factor, ftype='fir', zero_phase=True)

    def get_pulse_times(self):
        pulse_start_secs = [] # seconds
        pulse_end_secs = []   # seconds
        for i in range(0, len(self.pulses)):
            pulse_start_time_s, pulse_dfreq, pulse_peak_db, pulse_noise_floor_db, pulse_snr_db, pulse_duration_ms, pulse_sample_lenth, pulse_buffer_size, pulse_overlap, pulse_inter_ms = self.pulses[i].split(",")
            pulse_start_secs.append( float(pulse_start_time_s) )
            pulse_end_secs.append( float(pulse_start_time_s) + float(pulse_duration_ms) / 1e3 )
            if (self.max_db < float(pulse_peak_db)):
                self.max_db = float(pulse_peak_db)
            if (self.min_db > float(pulse_noise_floor_db)):
                self.min_db = float(pulse_noise_floor_db)
        
        return list(zip(pulse_start_secs, pulse_end_secs))

    def plot_pulse_fft(self, pulse_time, ax):
        pulse_start_sec = pulse_time[0] # seconds
        pulse_end_sec = pulse_time[1]# seconds
        pulse_start_sample = int(pulse_start_sec * self.sample_rate)
        pulse_end_sample = int(pulse_end_sec * self.sample_rate)
        pulse_iq = self.raw_iq[pulse_start_sample:pulse_end_sample]
        fft_freqs, fft_mags = self.pulse_fft(pulse_iq, masked = False)
        
        ax.plot(fft_freqs, fft_mags, label='Frequency magnitude')
        
        ax.set_title(f"Frequency-domain Magnitude")
        ax.grid(True)

    def plot_pulse_iq(self, pulse_time, ax):

        pulse_start_sec = pulse_time[0] # seconds
        pulse_end_sec = pulse_time[1]# seconds
        pulse_start_sample = int((pulse_start_sec - self.pulse_padding_s) * self.sample_rate)
        pulse_end_sample = int((pulse_end_sec + self.pulse_padding_s) * self.sample_rate)
        pulse_duration_samples = pulse_end_sample - pulse_start_sample

        print(f"Plotting pulse starting at {pulse_start_sec} s and ending at {pulse_end_sec} s.")
        pulse_duration_ms = (pulse_end_sec - pulse_start_sec) * 1e3

        pulse_mags = self.mags_smooth[pulse_start_sample:pulse_end_sample]
        pulse_slope = np.diff(pulse_mags) * 25

        # Plot time-domain I/Q
        ax.plot(self.raw_iq.real[pulse_start_sample:pulse_end_sample], label='I')
        ax.plot(self.raw_iq.imag[pulse_start_sample:pulse_end_sample], label='Q')
        ax.plot(pulse_mags, label='Magnitude')
        ax.plot(pulse_slope, label='Magnitude')

        # Mark the start/end of each pulse
        pulse_padding_samples = int(self.pulse_padding_s * self.sample_rate)
        ax.axvline( x = pulse_padding_samples, color = 'red' )
        ax.axvline( x = pulse_duration_samples - pulse_padding_samples, color = 'blue' )


        # plt.plot(I[burst_start_sample:burst_end_sample], label='I')
        # plt.plot(Q[burst_start_sample:burst_end_sample], label='Q')
        ax.set_title(f"Time-domain Magnitude ({pulse_duration_ms:.2f} ms)")
        ax.grid(True)


    def plot_spectrogram(self,  plt):

        # Plot spectrogram
        plt.figure(figsize=(10, 4))
        plt.specgram((self.raw_iq), NFFT=1024, Fs=self.sample_rate, noverlap=1, scale='dB', vmin=self.min_db - 100, vmax=self.max_db - 60)
        # plt.xlim(burst_start_sec,burst_end_sec)
        plt.ylim(-8e3,8e3)
        plt.title(f"Spectrogram for {len(data)} samples")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(label='dB')



    def reconstruct_gaussians(freqs, gmm):
        curves = []
        for i in range(gmm.n_components):
            mean = gmm.means_[i, 0]
            std = np.sqrt(gmm.covariances_[i, 0, 0])
            amp = 1.0  # or use gmm.weights_[i] if you want relative scaling

            curve = amp * np.exp(-0.5 * ((freqs - mean) / std)**2)
            curves.append(curve)
        return curves


    def extract_gmm_peaks(self, pulse_iq):
        # FFT and magnitude
        freqs, mags = self.pulse_fft(pulse_iq)

        if freqs is None:
            return []

        # Reshape for GMM
        mask = freqs >= 0
        X = freqs[mask].reshape(-1, 1)
        mags = mags[mask]


        # Find best fitting GMM to frequency distribution
        models = []
        for k in [1, 2]:
            gmm = GaussianMixture(n_components=k, random_state=0)
            gmm.fit(X)
            models.append((gmm, gmm.bic(X)))
        best_model = min(models, key=lambda x: x[1])[0]
        
        # Extract center frequencies and estimate magnitudes
        centers = best_model.means_.flatten()
        
        # Estimate magnitude at each center
        peak_mags = np.interp(centers, freqs, mags)

        return sorted(zip(centers, peak_mags), key=lambda x: -x[1])
        
    def pulse_fft(self, pulse_iq, masked = True):
        fft_samples = pulse_iq 
        if len(fft_samples) < self.min_fft_size: # Pad to min size
            fft_samples = np.pad(pulse_iq, (0, self.min_fft_size - len(fft_samples)))
        fft_size = len(fft_samples)
        fft_window = np.hanning( fft_size )
        fft = np.fft.fftshift(np.fft.fft(fft_samples * fft_window))
        fft_magnitude = np.abs(fft)
        fft_threshold = 0.1 * np.max(fft_magnitude) # 10% of peak
        fft_mask = fft_magnitude > fft_threshold
        if np.sum(fft_mask) == 0:
            return None, None
        else:
            freqs = np.fft.fftshift(np.fft.fftfreq( fft_size, d=1/self.sample_rate))
            if masked:
                return freqs[fft_mask], fft_magnitude[fft_mask]

            return freqs, fft_magnitude



if __name__ == "__main__":
    analysis = Pulse_Analysis()
