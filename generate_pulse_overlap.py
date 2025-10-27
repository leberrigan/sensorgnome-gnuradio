import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import detect_pulse as pd
import time
import detect_pulse_overlap as dpo



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

class generate_pulse_overlap:
    def __init__(self):

        self.first_pulse_idx = 0
        self.last_pulse_idx = 0
        self.pulse_overlap = 0.5
        self.freq_min = -1000 # Hz
        self.freq_max = 1000 # Hz

        self.pulse_padding_s = 2.5e-3
        self.min_fft_size = 64
        self.min_db = -20
        self.max_db = 20
        self.fft_masked = True
        self.step_width_hz = 50
        self.gaussian_width = 175  # Hz

        self.sample_rate = profiles[selected_profile]["sample_rate"]#/ downsample_factor
        self.raw_iq = np.fromfile(profiles[selected_profile]["filename"], dtype=np.complex64)
        # data = np.fromfile("../Data/raw iq/raw_iq_airspyhf_192000.0.bin", dtype=np.complex64)

        pulse_detector = pd.Pulse_Detector(samp_rate = self.sample_rate, verbose = True, min_snr_db=6, debounce_samples=10, pulse_len_ms=2.5, freq_min=self.freq_min, freq_max=self.freq_max)

        mags = np.abs(self.raw_iq)
        # Get moving average of magnitude
        self.mags_smooth = np.convolve(mags, np.ones(10)/10, mode='valid')

        time1 = time.time()
        self.pulses = pulse_detector.detect([self.mags_smooth, self.raw_iq])
        time2 = time.time()
        print(f"Pulse Detector time: {(time2 - time1):.3f} s vs sample time of {len(self.raw_iq) / self.sample_rate:.3f} s")


        if len(self.pulses) == 0:
            print("No pulses detected.")
            return

            
        pulse_times = self.get_pulse_times()
        
        n_pulses = len(self.pulses)

        # Overlap first with last
        combined_iq, pulse_1_iq, pulse_2_iq  = self.overlap_pulses(pulse_times[self.first_pulse_idx], pulse_times[self.last_pulse_idx], overlap = self.pulse_overlap)

        self.plot_pulse(combined_iq, pulse_1_iq, pulse_2_iq)


    def plot_pulse(self, combined_iq, pulse_1_iq, pulse_2_iq):

        figure, axes = plt.subplots(3, 2, figsize=(12, 3 * 3), sharex='col')
        plt.suptitle(f"Pulse overlap analysis (overlap {100*self.pulse_overlap:.0f}%)", fontsize = 18)

        # Magnitude
        times = np.arange(len(combined_iq)) / self.sample_rate
        axes[0,0].plot(times, 20 * np.log10(np.abs(pulse_1_iq) + 1e-12), color = "red")
        axes[1,0].plot(times, 20 * np.log10(np.abs(combined_iq) + 1e-12))
        axes[2,0].plot(times, 20 * np.log10(np.abs(pulse_2_iq) + 1e-12), color = "green")
        axes[0,0].set_title("Pulse 1")
        axes[1,0].set_title("Combined")
        axes[2,0].set_title("Pulse 2")
        axes[2,0].set_xlabel("Time (s)")
        axes[1,0].set_ylabel("Magnitude (dB)")

        # FFTs
        freqs_1, fft_mag_1 = self.pulse_fft(pulse_1_iq, masked = self.fft_masked)
        freqs_combined, fft_mag_combined = self.pulse_fft(combined_iq, masked = self.fft_masked)
        freqs_2, fft_mag_2 = self.pulse_fft(pulse_2_iq, masked = self.fft_masked)
        axes[0,1].plot(freqs_1 / 1e3, 20 * np.log10(fft_mag_1 + 1e-12), color = "red")
        axes[1,1].plot(freqs_combined / 1e3, 20 * np.log10(fft_mag_combined + 1e-12), color = "blue")
        axes[2,1].plot(freqs_2 / 1e3, 20 * np.log10(fft_mag_2 + 1e-12), color = "lime")
        axes[0,1].set_title("Pulse 1 FFT")
        axes[1,1].set_title("Combined FFT")
        axes[2,1].set_title("Pulse 2 FFT")
        axes[2,1].set_xlabel("Frequency (kHz)")
        axes[1,1].set_ylabel("Magnitude (dB)")

        
        pulse_actual_peaks = (freqs_1[np.argmax(fft_mag_1)], freqs_2[np.argmax(fft_mag_2)])
        
        fft_threshold = 0.5 * np.max(fft_mag_combined) # 10% of peak
        freqs_combined_masked = freqs_combined[fft_mag_combined > fft_threshold]
        print(f"Min freq:{min(freqs_combined_masked):.0f} Hz Max freq:{max(freqs_combined_masked):.0f} Hz")
        time1 = time.time()
        pulse_overlaps = dpo.Detect_Pulse_Overlap(freqs_combined, fft_mag_combined, freq_min=min(freqs_combined_masked), freq_max=max(freqs_combined_masked))
        gaussians = pulse_overlaps.analyze()
        # fits = self.get_gaussian_fits(freqs_combined, fft_mag_combined)
        # gaussians, centers = self.reconstruct_gaussians_from_fits(freqs_combined, fits, self.gaussian_width)
        time2 = time.time()

        print(f"Processing time for Gaussian fits: {(time2 - time1) * 1e3:.3f} ms ({len(combined_iq)} samples)")
        combined_curve = None
        
        for i, gaussian in enumerate(gaussians):
            curve, center = gaussian
            curve_db = 20 * np.log10(curve + 1e-12)  # avoid log(0)
            
            print(f"Pulse {i+1}: peak at {center:.3f} Hz with magnitude {max(curve):.3f}")
            axes[i*2,1].axvline( x = pulse_actual_peaks[i] / 1e3, color = ("red","lime")[i] )
            axes[1,1].plot(freqs_combined / 1e3, curve_db, linestyle='--', label=f'Gaussian {i+1}', color = ("darkred","darkgreen")[i])
            axes[i*2,1].plot(freqs_combined / 1e3, curve_db, linestyle='--', label=f'Gaussian {i+1}', color = ("darkred","darkgreen")[i])
            axes[i*2,1].axvline( x = center / 1e3, linestyle='--', color = ("darkred","darkgreen")[i] )
            axes[i*2,1].set_title(f"Pulse {i + 1} FFT ({np.abs(pulse_actual_peaks[i] - center):.1f} Hz diff)")
            if (i == 0):
                combined_curve = curve
            else:
                combined_curve += curve
        
        combined_curve_db = 20 * np.log10(combined_curve + 1e-12)
        axes[1,1].plot(freqs_combined / 1e3, combined_curve_db, linestyle='--', label=f'Gaussian {i+1}', color = "gray")

        for ax in axes[:,0]:
            ax.grid(True)

        for ax in axes[:,1]:
            ax.grid(True)
            ax.set_ylim(self.min_db,self.max_db+30)

        plt.tight_layout()
        plt.show()

    def overlap_pulses(self, pulse_1_times, pulse_2_times, overlap = 1):
        pulse_padding_samples = self.pulse_padding_s * self.sample_rate
        
        pulse_1_sample_start = self.sample_rate * pulse_1_times[0]
        pulse_1_sample_end = self.sample_rate * pulse_1_times[1]

        if (self.first_pulse_idx == self.last_pulse_idx):
            pulse_2_sample_start = pulse_1_sample_start
            pulse_2_sample_end = pulse_1_sample_end
            pulse_1_iq = pulse_2_iq = np.pad(self.raw_iq[int(pulse_1_sample_start):int(pulse_1_sample_end)], (int(pulse_padding_samples),int(pulse_padding_samples)), 'constant', constant_values=(0, 0))
            combined_iq = pulse_1_iq
        else:
            pulse_2_sample_start = self.sample_rate * pulse_2_times[0]
            pulse_2_sample_end = self.sample_rate * pulse_2_times[1]
        
            sample_shift = (1 - overlap) * (pulse_1_sample_end - pulse_1_sample_start)

            pulse_1_iq = np.pad(self.raw_iq[int(pulse_1_sample_start):int(pulse_1_sample_end)], (int(pulse_padding_samples),int(pulse_padding_samples)), 'constant', constant_values=(0, 0))
            pulse_2_iq = np.pad(self.raw_iq[int(pulse_2_sample_start):int(pulse_2_sample_end)], (int(pulse_padding_samples),int(pulse_padding_samples)), 'constant', constant_values=(0, 0))

            print(f"Length of pulse 1 {len(pulse_1_iq)} samples, pulse 2 {len(pulse_2_iq)} samples")

            if (len(pulse_1_iq) < len(pulse_2_iq)):
                pulse_1_iq = np.pad(pulse_1_iq, (0,len(pulse_2_iq) - len(pulse_1_iq)), 'constant', constant_values=(0, 0))
            elif (len(pulse_1_iq) > len(pulse_2_iq)):
                pulse_2_iq = np.pad(pulse_2_iq, (0,len(pulse_1_iq) - len(pulse_2_iq)), 'constant', constant_values=(0, 0))

            combined_iq = pulse_1_iq + pulse_2_iq

        print(f"Pulse 1: {pulse_1_times[2]} Hz, Pulse 2: {pulse_2_times[2]} Hz")
        # dfreq1 > dfreq2
        if (pulse_1_times[2] > pulse_2_times[2]):
            return combined_iq, pulse_2_iq, pulse_1_iq
        return combined_iq, pulse_1_iq, pulse_2_iq

    def get_pulse_times(self):
        pulse_start_secs = [] # seconds
        pulse_end_secs = []   # seconds
        pulse_dfreqs = []   # seconds
        for i in range(0, len(self.pulses)):
            pulse_start_time_s, pulse_dfreq, pulse_peak_db, pulse_noise_floor_db, pulse_snr_db, pulse_duration_ms, pulse_sample_lenth, pulse_buffer_size, pulse_overlap, pulse_inter_ms = self.pulses[i].split(",")
            pulse_start_secs.append( float(pulse_start_time_s) )
            pulse_end_secs.append( float(pulse_start_time_s) + float(pulse_duration_ms) / 1e3 )
            pulse_dfreqs.append( float(pulse_dfreq) )
            if (self.max_db < float(pulse_peak_db)):
                self.max_db = float(pulse_peak_db)
            if (self.min_db > float(pulse_noise_floor_db)):
                self.min_db = float(pulse_noise_floor_db)
        
        print("Min DB: ", self.min_db, " Max DB: ", self.max_db)
        return list(zip(pulse_start_secs, pulse_end_secs, pulse_dfreqs))
        

    def pulse_fft(self, pulse_iq, masked=True, overlap=0.75):
        if len(pulse_iq) < self.min_fft_size:
            window_size = self.min_fft_size
        else:
            window_size = len(pulse_iq)

        step_size = int(window_size * (1 - overlap))
        if step_size < 1:
            raise ValueError("Overlap too high — step size must be >= 1")

        # Pad if needed
        if len(pulse_iq) < window_size:
            pulse_iq = np.pad(pulse_iq, (0, window_size - len(pulse_iq)))

        n_windows = max(1, (len(pulse_iq) - window_size) // step_size + 1)
        fft_accum = []

        for i in range(n_windows):
            start = i * step_size
            end = start + window_size
            segment = pulse_iq[start:end]

            if len(segment) < window_size:
                segment = np.pad(segment, (0, window_size - len(segment)))

            windowed = segment * np.hanning(window_size)
            fft = np.fft.fftshift(np.fft.fft(windowed))
            fft_magnitude = np.abs(fft)
            fft_accum.append(fft_magnitude)

        # Combine spectra (average)
        fft_magnitude_avg = np.mean(fft_accum, axis=0)
        freqs = np.fft.fftshift(np.fft.fftfreq(window_size, d=1/self.sample_rate))

        # Apply threshold mask
        fft_threshold = 0.01 * np.max(fft_magnitude_avg)
        fft_mask = fft_magnitude_avg > fft_threshold

        if np.sum(fft_mask) == 0:
            return None, None

        if masked:
            return freqs[fft_mask], fft_magnitude_avg[fft_mask]
        else:
            return freqs, fft_magnitude_avg 
    """ 

    def get_gaussian_fits(self, freqs, mags):
        candidate_centers = self.get_candidate_centers(freqs)
        single_fits = self.fit_gaussians(freqs, mags, candidate_centers)
        double_fit_error, best_double = self.fit_gaussian_pairs(freqs, mags, candidate_centers)

        best_single = min(single_fits, key=lambda x: x[2])

        mu, A, _ = best_single
        G = np.exp(-0.5 * ((freqs - mu) / self.gaussian_width)**2)
        fit_single = A * G

        mu1, A1, mu2, A2, _ = best_double
        G1 = np.exp(-0.5 * ((freqs - mu1) / self.gaussian_width)**2)
        G2 = np.exp(-0.5 * ((freqs - mu2) / self.gaussian_width)**2)
        fit_double = A1 * G1 + A2 * G2

        if self.is_harmonic(mu1, mu2, A1, A2) or self.is_harmonic(mu2, mu1, A2, A1) or best_single[2] < double_fit_error or abs(mu1 - mu2) < self.gaussian_width * 2:
            return [best_single]
        else:
            return [best_double]


    def get_candidate_centers(self, freqs):
        step = self.step_width_hz  # Hz
        candidate_centers = np.arange(freqs.min(), freqs.max(), step)
        return candidate_centers

    def gaussian(self, f, mu, sigma=150):
        return np.exp(-0.5 * ((f - mu) / sigma)**2)
        
    def is_harmonic(self, f1, f2, A1, A2, tolerance=0.05):
        ratio = f2 / f1
        print("Checking harmonic: f1={:.1f}, f2={:.1f}, ratio={:.3f}".format(f1, f2, ratio))
        for n in range(2, 6):  # check up to 5th harmonic
            if abs(ratio - n) < tolerance and A2 < 0.5 * A1:
                return True
        return False


    def fit_gaussians(self, freqs, mags, candidate_centers):
        fits = []
        for mu in candidate_centers:
            G = self.gaussian(freqs, mu, self.gaussian_width)
            A = np.dot(mags, G) / np.dot(G, G)
            fit = A * G
            residual = np.linalg.norm(mags - fit)
            fits.append((mu, A, residual))
        return fits
    
    def fit_gaussian_pairs(self, freqs, mags, candidate_centers):
        from itertools import combinations

        best_double = None
        best_error = np.inf

        for mu1, mu2 in combinations(candidate_centers, 2):
            G1 = self.gaussian(freqs, mu1)
            G2 = self.gaussian(freqs, mu2)
            X = np.vstack([G1, G2]).T
            A, _, _, _ = np.linalg.lstsq(X, mags, rcond=None)
            fit = A[0] * G1 + A[1] * G2
            error = np.linalg.norm(mags - fit)
            if error < best_error:
                best_error = error
                best_double = (mu1, A[0], mu2, A[1], fit)

        return best_error, best_double

    def reconstruct_gaussians_from_fits(self, freqs, fits, width=150):
        curves = []
        centers = []
        for fit in fits:
            if len(fit) == 3:  # single Gaussian: (mu, A, residual)
                mu, A, _ = fit
                curve = A * np.exp(-0.5 * ((freqs - mu) / width)**2)
                curves.append(curve)
                centers = [mu]
            elif len(fit) == 5:  # double Gaussian: (mu1, A1, mu2, A2, fit_array)
                mu1, A1, mu2, A2, _ = fit
                curve1 = A1 * np.exp(-0.5 * ((freqs - mu1) / width)**2)
                curve2 = A2 * np.exp(-0.5 * ((freqs - mu2) / width)**2)
                curves.extend([curve1, curve2])
                centers = [mu1, mu2]
        return curves, centers
    """


if __name__ == "__main__":
    pulse_overlap = generate_pulse_overlap()
