"""

    Detect an overlap in pulses using gaussin fits

"""

import numpy as np
from sklearn.mixture import GaussianMixture
import detect_pulse as pd
import time
from itertools import combinations
from scipy import signal


class Detect_Pulse_Overlap:
    def __init__(self, fft_freqs, fft_mags, gaussian_width=175, step_width_hz = 50, fft_masked=True, freq_min=None, freq_max=None, sample_rate=None):
        self.fft_freqs = fft_freqs
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.fft_mags = fft_mags
        self.gaussian_width = gaussian_width # hz
        self.step_width_hz = step_width_hz
        self.fft_masked = fft_masked
        self.sample_rate = sample_rate
        self.min_fft_size = 64 # Samples
        self.gaussians = {}


    def analyze(self):
        # self.get_peaks(self.fft_mags)
        best_single = self.get_gaussian_fits(self.fft_freqs, self.fft_mags)
        curves, centers

        curves, centers = self.reconstruct_gaussians_from_fits(self.fft_freqs, fits, self.gaussian_width)
        return list(zip(curves, centers))

    def get_peaks(self, mags):
        peak_idxs, _ scipy.find_peaks(mags)
        if peak_indices.size == 0:
            peak_idxs = np.array([int(np.argmax(mags))])
        
        max_idx = int(peak_indices[np.argmax(mags[peak_indices])])
        amplitude_threshold = mags[max_idx] / (10 ** (10.0 / 20.0))
            
        selected_indices = [int(idx) for idx in peak_indices if mags[idx] >= amplitude_threshold]
        if max_idx not in selected_indices:
            selected_indices.append(max_idx)

        selected_indices = sorted(selected_indices, key=lambda idx: mags[idx], reverse=True)
        dominant_idx = selected_indices[0]
        dominant_freq_hz = float(freq_axis[dominant_idx])
        dominant_amplitude = float(magnitude_norm[dominant_idx])
        dominant_power_db = 20.0 * np.log10(dominant_amplitude + 1e-12)
        dominant_phase_rad = float(np.angle(spectrum[dominant_idx]))

        
    def get_gaussian_fits(self, freqs, mags):
        freq_min = freqs.min() if self.freq_min is None else self.freq_min
        freq_max = freqs.max() if self.freq_max is None else self.freq_max
        candidate_centers = self.get_candidate_centers(freqs, freq_min, freq_max)
        # print(f"Number of candidate centers between {freq_min:.0f} and {freq_max:.0f} Hz: {len(candidate_centers)}")
        single_fits = self.fit_gaussians(freqs, mags, candidate_centers)
        best_single = min(single_fits, key=lambda x: x[2])
        # mu, A, _ = best_single
        # G = np.exp(-0.5 * ((freqs - mu) / self.gaussian_width)**2)
        # fit_single = A * G
        return best_single


    def synthesize_tone(amplitude, frequency_hz, phase_rad, length, dtype):
        """Create a complex sinusoid with the supplied amplitude, frequency, and phase."""
        if length <= 0:
            return np.zeros(0, dtype=dtype)

        sample_indices = np.arange(length, dtype=np.float64)
        phase = phase_rad + 2.0 * np.pi * frequency_hz * sample_indices / self.sample_rate
        tone = amplitude * np.exp(1j * phase)
        return tone.astype(dtype, copy=False)
        
    def get_candidate_centers(self, freqs, freq_min = None, freq_max = None):
        step = self.step_width_hz  # Hz
        candidate_centers = np.arange(freq_min, freq_max, step)
        return candidate_centers

    def gaussian(self, f, mu, sigma=150):
        return np.exp(-0.5 * ((f - mu) / sigma)**2)
        
    def is_harmonic(self, f1, f2, A1, A2, tolerance=0.05):
        ratio = f2 / f1
        # print("Checking harmonic: f1={:.1f}, f2={:.1f}, ratio={:.3f}".format(f1, f2, ratio))
        for n in range(2, 6):  # check up to 5th harmonic
            if abs(ratio - n) < tolerance and A2 < 0.5 * A1:
                return True
        return False


    def fit_gaussians(self, freqs, mags, candidate_centers):
        fits = []
        for mu in candidate_centers:
            G = self.gaussians[mu] = self.gaussian(freqs, mu, self.gaussian_width)
            A = np.dot(mags, G) / np.dot(G, G)
            fit = A * G
            residual = np.linalg.norm(mags - fit)
            fits.append((mu, A, residual))
        return fits
    
    def fit_gaussian_pairs(self, freqs, mags, candidate_centers):

        best_double = None
        best_error = np.inf
        
        for mu1, mu2 in combinations(candidate_centers, 2):
            if abs(mu2 - mu1) < self.gaussian_width:
                continue
            G1 = self.gaussians[mu1]
            G2 = self.gaussians[mu2]
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