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

from gnuradio import gr

class blk(gr.sync_block):
    """
    Hysteresis + debounce comparator with sample-accurate timestamping.
    - Streaming output: float32 0.0/1.0
    - Prints per-pulse CSV to stdout
    """

    def __init__(self, output_type="file", verbose=True, filename="output.csv", port = 0, samp_rate=250000, min_snr_db=6, debounce_samples=10, pulse_len_ms = 2.5):
        gr.sync_block.__init__(self,
            name="Pulse detection logger",
            in_sig=[np.float32, np.complex64],  # mag, raw iq
            out_sig=[])
        
        self.filename = filename
        self.output_type = output_type
        self.verbose = verbose
        self.port = int(port)

        self.samp_rate = float(samp_rate)
        self.min_snr_db = int(min_snr_db)
        self.pulse_len_ms = float(pulse_len_ms)
        self.pulse_len_var = 0.5 # Allowed variability in pulse length (proportionally)
        self.debounce = int(debounce_samples)
        self.init_time = time.time()
        
        if self.output_type == "file":
            import csv
            self.csvfile = open(self.filename, "w", newline="")
            self.writer = csv.writer(self.csvfile)
            self.writer.writerow(["port","sample_index_start","ts_start","ts_end","duration_ms","duration_samples","peak_db", "noise_db","phase_jumps", "dphase", "dfreq", "inter_ms"]) # header
        if self.verbose:
            print("port, sample_index_start, ts_start, ts_end, duration_ms, duration_samples, peak_db, noise_db, phase_jumps, dphase, dfreq, inter_ms") # header

        # internal state
        self.time_chunk_start = None
        self.min_fft_size = 64
        self._state = 0
        self._count = 0
        self.sample_counter = 0
        self.sample_index_start = None
        self.sample_index_end = None
        self.start_time_s = None
        self.end_time_s = None
        self.duration_ms = None
        self.prev_start_time_s = None
        self.inter_ms = 0
        self.noise_floor = None
        self.pulse_iq = []

    """
        The 'work' function runs once per X samples, where X is the chunk size that gnuradio determines based on processing capacity. It's usually around 2048 samples, but can be lower.

        We expect it to be common for pulses to occur on the boundary of two chunks so we want to maintain pulse edge detection between iterations

        Time is calculated by

    """
    def work(self, input_items, output_items):
        mag = input_items[0]
        raw_iq = input_items[1]
        pulse_found = False
        # out = output_items[0]
        n = len(mag) # Number of input items

        # Only get noise floor at the beginning if not already set
        if self.noise_floor is None:
            self.noise_floor = np.percentile(mag, 90) # 90th percentile as noise floor estimate (works even if there are 100x transmitters emitting 4x 2.5ms pulses every 10 seconds and there is no overlap)
        noise_floor_db = 10.0 * np.log10( self.noise_floor )

        # Calculate the threshold magnitudes for the rising and falling edge of a pulse
        rise_thresh = max( 10 ** ((noise_floor_db + self.min_snr_db) / 10), 1e-12)
        fall_thresh = max( 10 ** ((noise_floor_db + self.min_snr_db / 2) / 10), 1e-12)

        # Update the clock once per chunk
        self.time_chunk_start = time.time()

        if n == 0:
            return 0

        for i in range(n):
            x = float( mag[i] )
            if self._state == 0: # Detecting pulse, looking for rising edge
                if x >= rise_thresh: # Potential rising edge
                    self._count += 1
                    if (self._count == 1):
                        self.start_idx = self.sample_counter + i
                        self.start_time_s = self.time_chunk_start + i / self.samp_rate
                    if self._count >= self.debounce: # Confirmed rising edge
                        self._state = 1
                        self.pulse_iq.append( raw_iq[i] )
                        self. _count = 0
                else: # Not a rising edge
                    self._count = 0
            elif self._state == 1: # Detecting pulse, looking for falling edge
                self.pulse_iq.append( raw_iq[i] )
                if x < fall_thresh: # Potential falling edge
                    self._count += 1
                    if (self._count == 1):
                        self.end_idx = self.sample_counter + i
                        self.end_time_s = self.time_chunk_start + i / self.samp_rate
                    if self._count >= self.debounce: # Confirmed falling edge
                        self._count = 0
                        duration_samples = self.end_idx - self.start_idx
                        self.duration_ms = round((duration_samples / self.samp_rate) * 1000.0, 2)
                        if self.verbose and ( self.duration_ms < self.pulse_len_ms * (1.0 - self.pulse_len_var) or self.duration_ms > self.pulse_len_ms * (2.0 + self.pulse_len_var * 2) ): # Pulse length is outside expected range
                            print(f"Discarding pulse of {self.duration_ms} ms with noise floor {noise_floor_db} and {duration_samples} samples collected. Chunk size of {n}", flush=True)
                            self._state = 3 # Start over
                        else: # Confirmed pulse length is within expected range
                            self._state = 2
                else: # Not a falling edge
                    self._count = 0
            elif self._state == 2: # Write the pulse to the CSV file

                pulse_start = self.start_idx - self.sample_counter
                pulse_end = self.end_idx - self.sample_counter
                
                # self.end_time_s = self.init_time + self.end_idx / self.samp_rate

                if self.prev_start_time_s is not None:
                    self.inter_ms = round((self.start_time_s - self.prev_start_time_s) * 1000.0, 2)

                pulse_mag = np.abs(self.pulse_iq)


                # Get the peak signal strength
                peak_magnitude = np.max(pulse_mag)
                peak_db = 10.0 * np.log10( peak_magnitude )

                # Get the SNR
                pulse_snr_db = (10 * np.log10( np.mean( pulse_mag ) ) ) - noise_floor_db

                # Get the IQ of the pulse
                self.pulse_iq = np.array(self.pulse_iq, dtype=np.complex64)


                dfreq = self.estimate_dfreq_phasor()
                dphase = self.estimate_phase_offset( peak_magnitude, pulse_mag, dfreq)
                """ 
                if self.verbose and self.duration_ms > self.pulse_len_ms * (1.0 + self.pulse_len_var) and phase_jumps == 0:
                    print(f"Discarding long pulse of {self.duration_ms} ms but with 0 phase jumps. Peak signal strength of {peak_db} and noise floor {noise_floor_db} with {len(self.pulse_iq)}. Chunk size of {n}", flush=True)
                    self._state = 3 # Start over
                    continue
                """
                pulse_found = True
                
                if self.verbose:
                    print(f"Pulse detected: {self.port} {self.start_time_s:.6f}, {self.duration_ms}, {len(self.pulse_iq)}, {peak_db:.2f}, {noise_floor_db:.2f}, {pulse_snr_db:.2f}, {dphase:.4f}, {dfreq / 1e3:.3f}, {self.inter_ms}. Chunk size of {n}", flush=True)
                elif self.output_type == "stream":
                    print(f"p{self.port},{self.start_time_s:.6f},{(dfreq / 1e3):.3f},{peak_db:.2f},{noise_floor_db:.2f},{pulse_snr_db:.2f},{self.duration_ms},{len(self.pulse_iq)},{dphase:.4f},{n},{self.inter_ms}", flush=True)
                
                if hasattr(self, 'csvfile'):
                    self.writer.writerow([self.port, self.start_idx, round(self.start_time_s, 6), round(self.end_time_s, 6), self.duration_ms, len(self.pulse_iq), round(peak_db,2), round(noise_floor_db,2), round(dphase, 4), round(dfreq / 1e3, 3), self.inter_ms])

                # Calculate time since last pulse
                self.prev_start_time_s = self.start_time_s 
                # Reset vars
                self._state = 3
            else: # state == 3 -> start over
                self._state = 0
                self.pulse_iq = []

        
        if not pulse_found and self._state == 0: # Update noise floor only if no pulse was found in this chunk
            self.noise_floor = np.percentile(mag, 90)

        if hasattr(self, 'csvfile'):
            self.csvfile.flush()

        self.sample_counter += n

        return n

    def estimate_dfreq_phasor(self):
        """
        Time-domain phasor estimate (arg of averaged phasor).
        Returns (dfreq_hz, rsd, mean_phasor).
          - dfreq_hz : estimated frequency offset in Hz
        """
        x = np.asarray(self.pulse_iq)
        if x.size < 2:
            return 0.0, np.nan, 0+0j

        d = x[1:] * np.conjugate(x[:-1])  # per-sample phasors (complex increments)
        mean_phasor = np.mean(d)
        mean_delta_phi = np.angle(mean_phasor)  # radians/sample

        dfreq_hz = (mean_delta_phi / (2.0 * np.pi)) * float(self.samp_rate)
        rsd = np.sqrt(max(0.0, 1.0 - np.abs(mean_phasor)))
        return float(dfreq_hz)

    def get_phase_at_midpoint(self, dfreq):
        N = len(self.pulse_iq)
        if N < 3:
            return 0.0, 0.0 # Not enough samples
    
        # Time vector for correction
        t = np.arange(N) / self.samp_rate
        correction = np.exp(-1j * 2 * np.pi * dfreq * t)
        iq_corrected = self.pulse_iq * correction

        # Unwrap phase
        phase_unwrapped = np.unwrap(np.angle(iq_corrected))

        # Midpoint index
        mid = N // 2
        phase_angle = phase_unwrapped[mid]
        
        # Central difference for slope
        dphi = phase_unwrapped[mid + 1] - phase_unwrapped[mid - 1]
        dt = 2 / self.samp_rate
        phase_slope = dphi / dt

        return phase_angle, phase_slope

    def matched_filter_alignment(self, tone_freq=4000, cycles=6):
        """
        Aligns a 4 kHz tone to a pulse using matched filtering.
        
        Returns:
            phase_offset (float): Phase offset (radians) between signal and reference tone.
        """
        # Generate reference tone
        t = np.arange(int(self.samp_rate * cycles / tone_freq)) / self.samp_rate
        ref_tone = np.exp(1j * 2 * np.pi * tone_freq * t)

        # Apply matched filter (complex correlation)
        corr = correlate(self.pulse_iq, ref_tone, mode='valid')
        peak_index = np.argmax(np.abs(corr))
        
        # Phase offset at peak
        phase_offset = np.angle(corr[peak_index])

        return  phase_offset

    def estimate_phase_offset(self, peak_magnitude, pulse_mag, dfreq):
        
        # Calculate phase offset from ideal sinewave at same frequency
        phase_threshold = 0.2 * peak_magnitude
        phase_mask = pulse_mag > phase_threshold
        N = len(self.pulse_iq)
        if N == 0:
            return 0.0

        if len(self.pulse_iq[phase_mask]) == 0 or np.any(np.isnan(self.pulse_iq)) or np.any(np.isinf(self.pulse_iq)):
            phase_offset = 0.0
            phase_jumps = 0
        else:                                        

            phase_idxs = np.arange(len(self.pulse_iq[phase_mask]))
            phase_correction = np.exp(-1j * 2 * np.pi * dfreq * N / self.samp_rate)
            pulse_iq_corrected = self.pulse_iq[phase_mask] * phase_correction
            phase_offset = np.median(np.angle(pulse_iq_corrected))

        return phase_offset

    def estimate_dfreq_fine_fft(self, samples: np.ndarray, sample_rate: float) -> float:
        """
        Estimate dfreq in kHz using fine FFT on complex pulse samples.
        """
        N = len(samples)
        if N == 0:
            return 0.0

        # Apply window
        window = np.hanning(N)
        windowed = samples * window

        # Perform FFT
        fft = np.fft.fft(windowed)
        fft_mag = np.abs(fft)

        # Find peak bin
        peak_bin = np.argmax(fft_mag)

        # Frequency axis
        freqs = np.fft.fftfreq(N, d=1/sample_rate)

        # Optional: sub-bin interpolation (quadratic)
        if 1 <= peak_bin < N - 1:
            alpha = fft_mag[peak_bin - 1]
            beta = fft_mag[peak_bin]
            gamma = fft_mag[peak_bin + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
            peak_freq = freqs[peak_bin] + p * (sample_rate / N)
        else:
            peak_freq = freqs[peak_bin]

        # Convert to kHz
        return peak_freq / 1000

    def estimate_dfreq_fft_from_poly(self):
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

        return freq_interp 
        
    def __del__(self):
        if hasattr(self, 'csvfile'):
            
            try:
                self.csvfile.close()
            except:
                pass