# GRC Python Block body (sync_block)
# Inputs: 3 (float32 energy, float32 phase, complex64 raw i/q)
# Outputs: 0
# Parameters: samp_rate, rise_thresh, fall_thresh, debounce_samples
# Lucas Berrigan 2025
# 
# This block detects pulses in a stream of energy values using hysteresis and debounce logic.
# It replaces the pulse detector using vamp-alsa-host
#
# Enhancements:
#  - Counts number of phase jumps in the pulse
# 
# Unknowns:
# - What is the allowable slop in pulse duration?
# 
# 
# 

import csv
import numpy as np
import time
from sklearn.mixture import GaussianMixture

class Pulse_Detector:
    """
        Detect pulses of a given duration with a minimum snr.

        Developed for the SensorGnome project for the Motus Wildlife Tracking System.

        Input:

        - Streams: 
            - IQ samples (np.complex64)
            - Moving average of signal magnitude
        - Arguments:
            - output_type: (Str'; default = "file") either "file" or "stream"
                - "file": output data to a CSV with name [filename]
                - "stream": print data in SensorGnome-compatible format to be captured on stdout
            - verbose: (Bool; default = True) to print or not to print message, that is the question
            - filename: (Str; default = "output.csv") name of csv file to output to
            - port: (Int; default = 0) USB port number to append to all messages returned when streaming
            - samp_rate: (Int; default = 250000) Sample rate of input stream IQ samples
            - min_snr_db: (Int; default = 6) Minimum SNR needed to detect a pulse, in dB
            - debounce_samples: (Int; default = 10) minimum samples needed to identify the start or end of a pulse. 
            - pulse_len_ms: (Float; default = 2.5) expected duration of a pulse, in ms

    """

    def __init__(self, output_type="test", verbose=True, filename="output.csv", port = 0, samp_rate=250000, min_snr_db=6, debounce_samples: int=10, pulse_len_ms: float=2.5):

        self.filename = filename
        self.output_type = output_type
        self.verbose = verbose
        self.port = int(port)

        self.samp_rate = int(samp_rate)
        self.min_snr_db = int(min_snr_db)
        self.pulse_len_ms = float(pulse_len_ms)
        self.pulse_len_var = 0.5 # Allowed variability in pulse length (proportionally)
        self.pulse_len_minimum = self.pulse_len_ms * (1.0 - self.pulse_len_var) # Minimum pulse length
        self.pulse_len_maximum = self.pulse_len_ms * (2.0 + self.pulse_len_var * 2) # Max pulse length

        self.debounce = int(debounce_samples)
        self.time_init = time.time()
        
        if self.output_type == "file":
            import csv
            self.csvfile = open(self.filename, "w", newline="")
            self.writer = csv.writer(self.csvfile)
            self.writer.writerow(["port","sample_index_start","ts_start","ts_end","duration_ms","duration_samples","peak_db", "noise_db","phase_jumps", "dphase", "dfreq", "inter_ms"]) # header
        if self.verbose:
            print("port, sample_index_start, ts_start, ts_end, duration_ms, duration_samples, peak_db, noise_db, phase_jumps, dphase, dfreq, inter_ms") # header

        # internal state
        self.min_fft_size = 64
        self._state = 0
        self._count = 0
        self.sample_counter = 0
        self.pulse_iq_start_s = None
        self.prev_start_time_s = None
        self.noise_floor = None
        self.pulse_iq = []
        self.reference_time = None
        self.time_chunk_start = None
        self.time_error_threshold = 0.1 # seconds
        
        """
            The 'work' function runs once per X samples, where X is the buffer size that gnuradio determines based on processing capacity. It's usually around 2048 samples, but can be lower.

            We expect it to be common for pulses to occur on the boundary of two buffers so we want to maintain pulse edge detection between iterations

            Time is calculated by

        """
    def detect(self, input):
        mag = input[0]
        raw_iq = input[1]
        pulse_found = False
        output = []
        # out = output_items[0]
        n = len(mag) # Number of input items
        if self.reference_time is None:
            self.reference_time = time.time()
        
        self.time_chunk_start = self.reference_time + self.sample_counter / self.samp_rate
        
        time_system = time.time()
        
        time_error = time_system - self.time_chunk_start

        if (time_error > self.time_error_threshold): 
            if self.verbose:
                print( f"Time error: {(time_error) * 1e3:.3f} ms" )
            self.sample_counter = 0
            self.reference_time = time.time()

        # Only get noise floor at the beginning if not already set
        if self.noise_floor is None:
            self.noise_floor = np.percentile(mag, 90) # 90th percentile as noise floor estimate (works even if there are 100x transmitters emitting 4x 2.5ms pulses every 10 seconds and there is no overlap)
        noise_floor_db = 10.0 * np.log10( self.noise_floor )

        # Calculate the threshold magnitudes for the rising and falling edge of a pulse
        rise_thresh = max( 10 ** ((noise_floor_db + self.min_snr_db) / 10), 1e-12)
        fall_thresh = max( 10 ** ((noise_floor_db + self.min_snr_db / 2) / 10), 1e-12)

        inter_ms = 0

        if n == 0:
            return 0
        
        for i in range(n):
            x = float( mag[i] )
            if self._state == 0: # Detecting pulse, looking for rising edge
                if x >= rise_thresh: # Potential rising edge
                    self._count += 1
                    self.pulse_iq.append( raw_iq[i] )
                    if (self._count == 1):
                        self.pulse_iq_start_s = self.time_chunk_start + i / self.samp_rate
                    if self._count >= self.debounce: # Confirmed rising edge
                        self._state = 1
                        self._count = 0
                elif self._count > 0: # Not a rising edge
                    self._count = 0
                    self._state = 3 # Start over
            elif self._state == 1: # Detecting pulse, looking for falling edge
                self.pulse_iq.append( raw_iq[i] )
                if x < fall_thresh: # Potential falling edge
                    self._count += 1
                    if self._count >= self.debounce: # Confirmed falling edge
                        self._count = 0
                        self._state = 2
                elif self._count > 0: # Not a rising edge
                    self._count = 0
            elif self._state == 2: # Write the pulse to the CSV file
                
                # Pulse magnitude
                pulse_mag = np.abs(self.pulse_iq)
                # Get rid of small ripples in the magnitude with moving average
                pulse_mag_smooth = np.convolve(pulse_mag, np.ones(10)/10, mode='valid')
                # Get the actual start and end times of the pulse by getting the max and min slopes of the magnitude
                pulse_slope = np.diff( pulse_mag_smooth )
                middle_idx = int(len(pulse_slope) / 2)
                start_idx = np.argmax( pulse_slope[:middle_idx] )
                end_idx = middle_idx + np.argmin( pulse_slope[middle_idx:] )

                start_time_s_0 = self.pulse_iq_start_s + start_idx / self.samp_rate
                end_time_s_0 = self.pulse_iq_start_s + end_idx / self.samp_rate
                duration_ms_0 = round((end_time_s_0 - start_time_s_0) * 1000.0, 2)
                # duration_ms = duration_ms_0
                start_time_s = self.pulse_iq_start_s + 0 / self.samp_rate
                end_time_s = self.pulse_iq_start_s + len(self.pulse_iq) / self.samp_rate

                duration_ms = round((end_time_s - start_time_s) * 1000.0, 2)
                
                if ( duration_ms < self.pulse_len_minimum or duration_ms > self.pulse_len_maximum ): # Pulse length is outside expected range
                    if self.verbose:
                        print(f"Pulse discarded due to duration mismatch ({duration_ms} ms): {self.pulse_iq_start_s}, {self.pulse_iq_start_s - self.time_chunk_start}, {start_idx}, {end_idx},{start_time_s - self.time_chunk_start:.6f},{noise_floor_db:.2f},{len(self.pulse_iq)},{n}") 
                    # if self.output_type == "test":
                    #     output.append(f"{start_time_s - self.time_chunk_start:.6f},0,0,{noise_floor_db:.2f},0,{duration_ms},{len(self.pulse_iq)},{n},0,0") 
                    self._state = 3 # Start over
                    continue 

                # Get the pulse interval
                if self.prev_start_time_s is not None:
                    inter_ms = round((start_time_s - self.prev_start_time_s) * 1000.0, 2) # Time since last pulse

                # Get the peak signal strength
                peak_magnitude = np.max(pulse_mag_smooth)
                peak_db = 10.0 * np.log10( peak_magnitude )

                # Get the SNR
                pulse_snr_db = (10 * np.log10( np.mean( pulse_mag_smooth ) ) ) - noise_floor_db

                # Get the corrected IQ of the pulse
                self.pulse_iq = np.array(self.pulse_iq[start_idx:end_idx], dtype=np.complex64)


                dfreq_centroid = self.estimate_dfreq_centroid()
                dfreq_median, dfreq_interp = self.estimate_dfreq_quadratic()
                dfreq_phasor = self.estimate_dfreq_phasor()
                dfreq_cubic = self.estimate_dfreq_fft_cubic()

                if False and self.output_type == "test":
                    pulse_peaks = self.extract_gmm_peaks()
                    if len(pulse_peaks) > 0:
                        for peak_freq, peak_mag in pulse_peaks:
                            print(f"Peak at {peak_freq:.3f} Hz with magnitude {peak_mag}")
                    
                
                pulse_found = True
                
                if self.verbose:
                    #print(f"Pulse detected: {self.port} {start_time_s:.6f}, {duration_ms}, {len(self.pulse_iq)}, {peak_db:.2f}, {noise_floor_db:.2f}, {pulse_snr_db:.2f}, {dfreq_cubic / 1e3:.3f}, {inter_ms}. Chunk size of {n}", flush=True)
                    # print(f"Frequency offsets: {dfreq_centroid / 1e3:.3f} {dfreq_median / 1e3:.3f} {dfreq_interp / 1e3:.3f} {dfreq_phasor / 1e3:.3f} {dfreq_cubic / 1e3:.3f} {inter_ms}")
                    print(f"{self.pulse_iq_start_s}, {self.pulse_iq_start_s - self.time_chunk_start}, {start_idx}, {end_idx},{start_time_s - self.time_chunk_start:.6f},{(dfreq_cubic / 1e3):.3f},{peak_db:.2f},{noise_floor_db:.2f},{pulse_snr_db:.2f},{duration_ms},{len(self.pulse_iq)},{n},{(dfreq_phasor / 1e3):.3f},{inter_ms}") 
                
                if self.output_type == "test":
                    output.append(f"{start_time_s_0 - self.time_chunk_start:.6f},{(dfreq_cubic / 1e3):.3f},{peak_db:.2f},{noise_floor_db:.2f},{pulse_snr_db:.2f},{duration_ms_0},{len(self.pulse_iq)},{n},{(dfreq_phasor / 1e3):.3f},{inter_ms}") 
                elif self.output_type == "stream":
                    print(f"p{self.port},{start_time_s:.6f},{(dfreq_cubic / 1e3):.3f},{peak_db:.2f},{noise_floor_db:.2f},{pulse_snr_db:.2f},{duration_ms},{len(self.pulse_iq)},{n},{(dfreq_phasor / 1e3):.3f},{inter_ms}", flush=True)
                
                if hasattr(self, 'csvfile'):
                    self.writer.writerow([self.port, start_idx, round(start_time_s, 6), round(end_time_s, 6), duration_ms, len(self.pulse_iq), round(peak_db,2), round(noise_floor_db,2), round(dfreq_cubic / 1e3, 3), inter_ms])

                # Calculate time since last pulse
                self.prev_start_time_s = start_time_s 
                # Reset vars
                self._state = 3
            else: # state == 3 -> start over
                self._state = 0
                self.pulse_iq = []

        # Update the clock once per chunk
        time_estimated = self.time_chunk_start + n / self.samp_rate
        time_system = time.time()
        time_error = time_system - self.time_chunk_start
        self.time_chunk_start = time_estimated + time_error * 0.001 # Slight correction for clock drift

        if not pulse_found and self._state == 0: # Update noise floor only if no pulse was found in this chunk
            self.noise_floor = np.percentile(mag, 90)

        if hasattr(self, 'csvfile'):
            self.csvfile.flush()

        if (self.output_type == "test"):
            return output
        
        self.sample_counter += n

        return n

    def extract_gmm_peaks(self):
        # FFT and magnitude
        freqs, mags = self.pulse_fft()

        if freqs is None:
            return []

        # Reshape for GMM
        mask = freqs >= 0
        X = freqs[mask].reshape(-1, 1)
        mags = mags[mask]


        # Find best fitting GMM to frequency distribution
        models = []
        if (len(freqs[mask]) < 2):
            return []

        for k in [1, 2]:
            gmm = GaussianMixture(n_components=k, random_state=0)
            gmm.fit(X)
            models.append((gmm, gmm.bic(X)))
        best_model = min(models, key=lambda x: x[1])[0]

        # Extract center frequencies and estimate magnitudes
        centers = best_model.means_.flatten()
        
        # Estimate magnitude at each center
        peak_mags = np.interp(centers, freqs[mask], mags)

        return sorted(zip(centers, peak_mags), key=lambda x: -x[1])
        
    def pulse_fft(self):
        fft_samples = self.pulse_iq 
        if len(fft_samples) < self.min_fft_size: # Pad to min size
            fft_samples = np.pad(self.pulse_iq, (0, self.min_fft_size - len(fft_samples)))
        fft_size = len(fft_samples)
        fft_window = np.hanning( fft_size )
        fft = np.fft.fftshift(np.fft.fft(fft_samples * fft_window))
        fft_magnitude = np.abs(fft)
        fft_threshold = 0.1 * np.max(fft_magnitude) # 10% of peak
        fft_mask = fft_magnitude > fft_threshold
        if np.sum(fft_mask) == 0:
            return None, None
        else:
            freqs = np.fft.fftshift(np.fft.fftfreq( fft_size, d=1/self.samp_rate))
            return freqs[fft_mask], fft_magnitude[fft_mask]

    def estimate_dfreq_centroid(self):
        fft_samples = self.pulse_iq
        if len(fft_samples) < self.min_fft_size:
            fft_samples = np.pad(self.pulse_iq, (0, self.min_fft_size - len(self.pulse_iq)))

        fft_size = len(fft_samples)
        fft_window = np.hanning(fft_size)
        fft = np.fft.fftshift(np.fft.fft(fft_samples * fft_window))
        fft_magnitude = np.abs(fft)

        # Frequency axis
        freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1/self.samp_rate))

        # Apply threshold mask (optional)
        fft_threshold = 0.1 * np.max(fft_magnitude)
        fft_mask = fft_magnitude > fft_threshold

        if np.sum(fft_mask) == 0:
            freq_centroid = 0.0
        else:
            # Spectral centroid: weighted average of frequencies
            freq_centroid = np.average(freqs[fft_mask], weights=fft_magnitude[fft_mask])

        return freq_centroid


    def estimate_dfreq_quadratic(self):
        fft_samples = self.pulse_iq 
        if len(fft_samples) < self.min_fft_size: # Pad to min size
            fft_samples = np.pad(self.pulse_iq, (0, self.min_fft_size - len(fft_samples)))
        fft_size = len(fft_samples)
        fft_window = np.hanning( fft_size )
        fft = np.fft.fftshift(np.fft.fft(fft_samples * fft_window))
        fft_magnitude = np.abs(fft)
        fft_threshold = 0.1 * np.max(fft_magnitude) # 10% of peak
        fft_mask = fft_magnitude > fft_threshold
        if np.sum(fft_mask) == 0:
            freq_interp = 0.0
        else:
            freqs = np.fft.fftshift(np.fft.fftfreq( fft_size, d=1/self.samp_rate))
            freq_median = float( np.median(freqs[fft_mask]) )   
            freq_peak = np.argmax(fft_magnitude)
            if 0 < freq_peak < len(fft_magnitude)-1:
                y0, y1, y2 = fft_magnitude[freq_peak-1:freq_peak+2]
                p = (y2 - y0) / (2*(2*y1 - y2 - y0))
                freq_interp = freqs[freq_peak] + p * (freqs[1] - freqs[0])

        return freq_median, freq_interp

    def estimate_dfreq_phasor(self):
        """
        Time-domain phasor estimate (arg of averaged phasor).
        Returns dfreq_hz : estimated frequency offset in Hz
        """
        x = np.asarray(self.pulse_iq)
        if x.size < 2:
            return 0.0, np.nan, 0+0j

        d = x[1:] * np.conjugate(x[:-1])  # per-sample phasors (complex increments)
        mean_phasor = np.mean(d)
        mean_delta_phi = np.angle(mean_phasor)  # radians/sample

        dfreq_hz = (mean_delta_phi / (2.0 * np.pi)) * float(self.samp_rate)
        
        return float(dfreq_hz)

    def estimate_dfreq_fft_cubic(self, zero_pad=True):
        """
        FFT-based dfreq estimate using Hamming window and cubic interpolation (matches repo).
        Returns (dfreq_hz, bin_est, peak_bin, peak_mag).
          - dfreq_hz : estimated frequency in Hz (positive or negative)
          - bin_est   : fractional bin index (already mapped so >N/2 -> negative)
          - peak_bin  : integer bin index of peak prior to interpolation
          - peak_mag  : magnitude at peak_bin
        """
        x = np.asarray(self.pulse_iq, dtype=np.complex64)
        L = x.size
        if L == 0:
            return 0.0, 0.0, None, 0.0

        # Hamming window
        w = self._hamming_window(L)
        xw = x * w

        # choose FFT length (m_frames-like): zero-pad to next power-of-two >= L if requested,
        # else use L (this emulates zero-padding behavior used in many FFT workflows).
        if zero_pad:
            N = self._next_pow2_at_least(L)
            if N < L:
                N = L
        else:
            N = L

        # zero-pad into length N
        if N > L:
            buf = np.zeros(N, dtype=np.complex64)
            buf[:L] = xw
        else:
            buf = xw

        # full complex FFT (so we can handle negative frequencies explicitly)
        X = np.fft.fft(buf)
        mags = np.abs(X)
        magsq = mags * mags

        peak_bin = int(np.argmax(mags))
        peak_mag = float(mags[peak_bin])

        # cubic sub-bin interpolation (match repo's FreqEstimator::estimateBinOffset)
        delta = self.estimate_bin_offset_cubic(magsq, peak_bin)
        bin_est = peak_bin + delta

        # map bins above Nyquist to negative frequencies (same as repo logic)
        if bin_est > N / 2.0:
            bin_est = - (N - bin_est)

        dfreq_hz = (bin_est / float(N)) * float(self.samp_rate)
        return float(dfreq_hz)

    def cubic_maximize(self, y0, y1, y2, y3):
        """
        Port of FreqEstimator::cubicMaximize from the repo.
        Inputs are function values at x = 0,1,2,3.
        Returns x (double) location of local cubic maximum (in same x units),
        or a sentinel < -900 on pathological failure, or -1 if 'a' == 0 (degenerate).
        """
        # convert to double precision for numerical stability
        y0 = float(y0); y1 = float(y1); y2 = float(y2); y3 = float(y3)

        a = y0 / -6.0 + y1 / 2.0 - y2 / 2.0 + y3 / 6.0
        if a == 0.0:
            return -1.0  # error/degenerate case (matches C++ return -1)

        b = y0 - 5.0 * y1 / 2.0 + 2.0 * y2 - y3 / 2.0
        c = -11.0 * y0 / 6.0 + 3.0 * y1 - 3.0 * y2 / 2.0 + y3 / 3.0

        # derivative coefficients (3ax^2 + 2bx + c) -> da x^2 + db x + dc where da=3a, db=2b, dc=c
        da = 3.0 * a
        db = 2.0 * b
        dc = c

        discriminant = db * db - 4.0 * da * dc
        if discriminant < 0.0:
            if discriminant < -1.0:
                return -1000.0  # error sentinel (as in C++)
            else:
                discriminant = 0.0

        sqrt_d = np.sqrt(discriminant)
        # two roots of derivative
        denom = 2.0 * da
        # guard denom (shouldn't be zero because we tested a != 0)
        x1 = (-db + sqrt_d) / denom
        x2 = (-db - sqrt_d) / denom

        # pick the one corresponding to a local maximum:
        # second derivative at x is 2*da*x + db (since derivative of derivative is 2*da*x + db)
        # choose x with negative second derivative
        if (2.0 * da * x1 + db) < 0.0:
            return x1
        else:
            return x2
    
    def estimate_bin_offset_cubic(self, magsq, k):
        """
        Given magnitude-squared spectrum array 'magsq' and integer peak index k,
        estimate sub-bin offset using cubic interpolation over four bins:
          y0 = magsq[k-1], y1 = magsq[k], y2 = magsq[k+1], y3 = magsq[k+2]
        The repo's cubicMaximize is defined for x = 0..3; to get offset relative to k
        we use delta = cubicMaximize(y0..y3) - 1.0
        Wrap indices modulo N to match repository WRAP_BIN behaviour.
        """
        N = len(magsq)
        if N == 0:
            return 0.0

        # wrapping helper
        def wrap(i):
            return (i + N) % N

        y0 = magsq[wrap(k - 1)]
        y1 = magsq[wrap(k + 0)]
        y2 = magsq[wrap(k + 1)]
        y3 = magsq[wrap(k + 2)]

        x = self.cubic_maximize(y0, y1, y2, y3)
        # handle error sentinels from cubic_maximize
        if x < -900.0:
            return 0.0
        if x == -1.0:
            return 0.0

        delta = x - 1.0  # because x is in [0..3] where 1 corresponds to the central bin
        return float(delta)

    def _hamming_window(self, N):
        if N <= 1:
            return np.ones(N, dtype=float)
        n = np.arange(N)
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (N - 1))

    def _next_pow2_at_least(self, x):
        if x <= 1:
            return 1
        return 1 << int(np.ceil(np.log2(x)))

    def __del__(self):
        if hasattr(self, 'csvfile'):
            
            try:
                self.csvfile.close()
            except:
                pass