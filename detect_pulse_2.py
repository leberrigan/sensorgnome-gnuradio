# GRC Python Block body (sync_block)
# Inputs: 3 (float32 energy, float32 phase, complex64 raw i/q)
# Outputs: 0
# Parameters: samp_rate, rise_thresh, fall_thresh, debounce_samples
# Lucas Berrigan 2025
# 
# This block detects pulses in a stream of IQ values using a combination of matched filter correlations and signal strength
# It replaces the pulse detector using vamp-alsa-host
#
# Enhancements:
#  - Counts number of phase jumps in the pulse
# 
# Unknowns:
# - What is the allowable slop in pulse duration?
# 
# 

"""

Prep:
    1. Make a rising edge template

Steps:

    1. Convolve signal with edge template and get set of correlations (detect_edges)
    2. Find peaks in the correlations with a minimum change in magnitude (consider_peaks)
    3. Do an FFT of the pulse window (computcompute_edge_ffte_)
    4. Find peaks in the FFT and synthesize tones corresponding to those peaks (apply_best_peak_subtraction)
    5. Subtract tones from pulse window, one at a time, to find the frequency which has the greatest 
        reduction in the overal magnitude over the pulse window (apply_best_peak_subtraction)
    6. Get the resulting IQ data with the peak subtracted and apply the continue looking for edges (handle_rising_edge)


"""

import csv
import numpy as np
import time
from scipy import signal
import detect_pulse_overlap as dpo


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

    def __init__(self, output_type="test", verbose=True, filename="output.csv", port = 0, samp_rate=250000, min_snr_db=6, debounce_samples: int=10, pulse_len_ms: float=2.5, pulse_padding_s: float=2.5e-3, freq_min: float=None, freq_max: float=None):

        self.filename = filename
        self.output_type = output_type
        self.verbose = verbose
        self.port = int(port)

        self.samp_rate = int(samp_rate)
        self.min_snr_db = 6#int(min_snr_db)
        self.magnitude_threshold_db = -40
        self.pulse_len_ms = float(pulse_len_ms)
        self.pulse_len_samples = int(pulse_len_ms * 1e-3 * samp_rate) # number of samples in a pulse
        self.pulse_len_var = 0.5 # Allowed variability in pulse length (proportionally)
        self.pulse_len_minimum = self.pulse_len_ms * (1.0 - self.pulse_len_var) # Minimum pulse length
        self.pulse_len_maximum = self.pulse_len_ms * (2.0 + self.pulse_len_var * 2) # Max pulse length
        self.pulse_padding_s = pulse_padding_s
        self.pulse_padding = int(pulse_padding_s * self.samp_rate) # Samples of padding to add to each side of the pulse for FFT analysis

        self.freq_min = freq_min
        self.freq_max = freq_max

        self.debounce = int(debounce_samples)
        self.time_init = time.time()
        
        self.pre_fft_results = 0

        if self.output_type == "file":
            import csv
            self.csvfile = open(self.filename, "w", newline="")
            self.writer = csv.writer(self.csvfile)
            self.writer.writerow(["port","sample_index_start","ts_start","ts_end","duration_ms","duration_samples","peak_db", "noise_db","phase_jumps", "dphase", "dfreq", "pulse_overlap", "inter_ms"]) # header
        if self.verbose:
            print("port, ts_start, ts_start_relative, sample_index_start, duration_samples, duration_ms, dfreq, peak_db, noise_db, snr_db, chunk_size, pulse_overlap, inter_ms") # header


        # internal state
        self.min_fft_size = 64
        self._state = 0
        self._count = 0
        self.sample_counter = 0
        self.pulse_iq_start_s = None
        self.prev_start_time_s = None
        self.noise_floor = None
        self.noise_buffer_len = int(1e3)
        self.noise_buffer = np.full(self.noise_buffer_len, np.nan)
        self.pulse_iq = []
        self.reference_time = None
        self.time_chunk_start = None
        self.time_error_threshold = 0.1 # seconds
        
        self.prev_chunk_end_samples = None
        self.pulse_iq_padding_start = []
        self.pulse_iq_padding_end = []
        self.pulse_iq_padded = None

        self.edge_duration_ms = 0.3
        self.edge_template = self.create_edge_template()
        self.edge_window_samples = max(1, int(self.edge_duration_ms * 1e-3 * self.samp_rate))
        self.envelope = self.get_template_envelope( self.pulse_len_samples, self.edge_window_samples )
        self.edge_correlation_threshold = 0.98
        self.edge_magnitude_threshold = 2e-4  # Minimum magnitude change to accept an edge
        self.fft_amplitude_threshold_db = 10 # relative dB from max
        self.prev_chunk_correlations = None

        self.prev_chunk_last_pulse_samples = None
        self.times = {
            "detect_edges": 0,
            "get_stdev": 0,
            "get_correlations1": 0,
            "get_correlations2": 0,
            "get_correlations3": 0,
            "pulse_detected": 0,
            "no_pulse_detected": 0,
            "get_baseline": 0,
            "get_nearby_correlations": 0,
            "get_edge_fft": 0,
            "subtract_signal": 0,
            "compute_fft": 0,
            "apply_best_peak": 0,
        }

        self.pulses = []

        """
            The 'work' function runs once per X samples, where X is the buffer size that gnuradio determines based on processing capacity. It's usually around 2048 samples, but can be lower.

            We expect it to be common for pulses to occur on the boundary of two buffers so we want to maintain pulse edge detection between iterations

            Time is calculated by

        """
    def detect(self, input):
        raw_iq = input[0].copy()
        pulse_found = False
        output = []
        timestamps = []
        # out = output_items[0]
        
        # Set clock
        if (self.output_type == "test"):
            self.reference_time = 0
        elif self.reference_time is None:
            self.reference_time = time.time()

        n = len(raw_iq) # Number of input items
        
        # Add pulse length samples to the start of the chunk
        if self.prev_chunk_end_samples is not None:
            raw_iq = np.concatenate([self.prev_chunk_end_samples, raw_iq]).astype(np.complex64)

        
        self.time_chunk_start = self.reference_time + self.sample_counter / self.samp_rate
        
        # Fix clock
        if (self.output_type != "test"):
            time_system = time.time()
            
            time_error = time_system - self.time_chunk_start

            if (time_error > self.time_error_threshold): 
                if self.verbose:
                    print( f"Time error: {(time_error) * 1e3:.3f} ms" )
                self.sample_counter = 0
                self.reference_time = time.time()
        
        tmp = time.time()
        # Detect edges
        rising_edges, correlations, subtracted_signal_iq, synthesized_signal_iq = self.detect_edges_matched_filter( raw_iq )
        self.times["detect_edges"] += time.time() - tmp

        if self.verbose:
            print(f"Found {len(rising_edges)} pulses in {len(correlations)} correlations!")
        if (self.output_type == "test"):
            output += rising_edges

        # Update the clock once per chunk
        if (self.output_type != "test"):
            time_estimated = self.time_chunk_start + n / self.samp_rate
            time_system = time.time()
            time_error = time_system - self.time_chunk_start
            self.time_chunk_start = time_estimated + time_error * 0.001 # Slight correction for clock drift

        # Store pulse length samples for the next chunk
        self.prev_chunk_end_samples = subtracted_signal_iq[-self.pulse_len_samples:]
        return_samples = subtracted_signal_iq[:-self.pulse_len_samples]


        if hasattr(self, 'csvfile'):
            self.csvfile.flush()

        self.sample_counter += len(return_samples)

        if (self.output_type == "test"):
            return output, correlations, return_samples, synthesized_signal_iq


        return n
    def extract_segments(self, signal_magnitude, template_length, start_idx, batch_size):
        signal_magnitude = np.asarray(signal_magnitude)
        end_idx = start_idx + batch_size

        # Ensure we don't go out of bounds
        max_start = len(signal_magnitude) - template_length
        if end_idx > max_start:
            end_idx = max_start

        # Compute all starting indices for this batch
        starts = np.arange(start_idx, end_idx)

        # Use broadcasting to extract segments efficiently
        idx_matrix = starts[:, None] + np.arange(template_length)[None, :]
        segments = signal_magnitude[idx_matrix]  # shape: (batch_size, template_length)

        return segments

    def detect_edges_matched_filter(self, signal_iq=None):
        """Detect rising edges using a matched filter with vectorised statistics.

        Hot-path cost reduced from O(N * template_length) to O(N + N log N):
          - scipy.signal.correlate computes all dot products in one FFT pass
          - prefix sums give O(1) window mean, variance, baseline, and plateau
          - only windows that overlap a subtracted pulse fall back to O(template_length)
        """
        signal_magnitude = np.abs(signal_iq)
        template_length = len(self.edge_template)
        signal_length = len(signal_magnitude)
        correlation_length = signal_length - template_length + 1

        if correlation_length <= 1:
            self.prev_chunk_correlations = None
            return [], np.zeros(max(0, correlation_length), dtype=float), signal_iq, np.zeros_like(signal_iq)

        third = max(1, template_length // 3)
        eps = 1e-12

        self.noise_floor = float(np.mean(signal_magnitude[:third]))
        synthesized_signal_iq = np.zeros(len(signal_iq), dtype=np.complex64)
        pulses = []

        # --- Precompute prefix sums on the initial signal_magnitude ---
        # prefix[k] = sum(signal_magnitude[0:k]), so window [a:b] sum = prefix[b] - prefix[a]
        prefix = np.empty(signal_length + 1)
        prefix[0] = 0.0
        np.cumsum(signal_magnitude, out=prefix[1:])
        prefix_sq = np.empty(signal_length + 1)
        prefix_sq[0] = 0.0
        np.cumsum(signal_magnitude ** 2, out=prefix_sq[1:])

        # --- Precompute raw cross-correlations ---
        # raw_xcorr[k] = dot(signal_magnitude[k : k+template_length], edge_template)
        # The edge_template is zero-mean, so the normalised correlation reduces to raw_xcorr / norm.
        tmp = time.time()
        raw_xcorr = signal.correlate(signal_magnitude, self.edge_template, mode='valid', method='fft')
        self.times["get_correlations1"] += time.time() - tmp

        correlations = np.zeros(correlation_length, dtype=float)
        if self.prev_chunk_correlations is not None:
            correlations[:len(self.prev_chunk_correlations)] = self.prev_chunk_correlations

        # dirty_end: signal_magnitude[0:dirty_end] may have been modified by a subtraction.
        # Windows whose start index is less than dirty_end are recomputed from signal_magnitude.
        dirty_end = 0
        # blocked_until: skip detection for iterations i < blocked_until to prevent
        # double-detection when apply_best_peak_subtraction returns None (no subtraction,
        # dirty_end not advanced) and the next iterations reuse stale raw_xcorr values.
        blocked_until = 0

        for i in range(1, correlation_length - 1):
            ws = i + 1                  # window start in signal_magnitude
            we = ws + template_length   # window end (exclusive)

            is_dirty = ws < dirty_end

            # --- O(1) window statistics (or O(template_length) fallback for dirty windows) ---
            if is_dirty:
                w = signal_magnitude[ws:we]
                w_sum = float(w.sum())
                w_sum_sq = float(np.dot(w, w))
                xcorr_val = float(np.dot(w, self.edge_template))
            else:
                w_sum = float(prefix[we] - prefix[ws])
                w_sum_sq = float(prefix_sq[we] - prefix_sq[ws])
                xcorr_val = float(raw_xcorr[ws])

            mean = w_sum / template_length
            variance = max(0.0, w_sum_sq / template_length - mean * mean)

            if variance < 1e-7:
                continue

            norm = np.sqrt(variance * template_length)
            # edge_template is zero-mean: dot(x - mean, t) = dot(x, t) since sum(t)=0
            correlations[i + 1] = xcorr_val / norm

            # --- Baseline (first third) and plateau (last third) ---
            if is_dirty:
                baseline = float(signal_magnitude[ws : ws + third].sum()) / third
                plateau = float(signal_magnitude[we - third : we].sum()) / third
            else:
                baseline = float(prefix[ws + third] - prefix[ws]) / third
                plateau = float(prefix[we] - prefix[we - third]) / third

            # --- Peak detection ---
            # Stores at [i+1] then checks [i] (computed last iteration) as "current".
            # This one-step delay allows the look-ahead comparison to work correctly.
            corr = correlations[i]          # peak candidate: set by previous iteration
            prev_corr = correlations[i - 1]
            next_corr = correlations[i + 1] # look-ahead: just computed above
            if corr <= prev_corr or corr <= next_corr or corr < self.edge_correlation_threshold:
                continue

            # Refractory block: skip this iteration if we just detected at i-1 or i-2.
            # Prevents double-detection when subtraction returned None and raw_xcorr is reused.
            if i < blocked_until:
                continue

            # Trailing-window suppression: reject if any earlier peak in the edge window was stronger.
            trailing_start = max(0, i - self.edge_window_samples)
            if corr <= np.max(correlations[trailing_start:i]):
                continue

            if plateau < baseline:
                continue

            magnitude_delta = plateau - baseline
            magnitude_delta_db = 10.0 * np.log10(max(magnitude_delta, eps))
            snr_db = 10.0 * np.log10(max(plateau / max(baseline, eps), eps))
            plateau_db = 10.0 * np.log10(max(plateau, eps))

            self.pre_fft_results += 1
            if magnitude_delta < self.edge_magnitude_threshold or snr_db < self.min_snr_db:
                continue

            edge_index = i + third
            edge_sample_index = max(0, min(int(round(edge_index)), signal_length - 1))
            edge_time = edge_index / self.samp_rate

            sv = signal_iq[edge_sample_index]
            edge_phase_rad = float(np.arctan2(sv.imag, sv.real))

            tmp = time.time()
            result = self.compute_edge_fft(signal_iq, edge_time)
            self.times["get_edge_fft"] += time.time() - tmp
            if result is None:
                continue

            tmp = time.time()
            result["edge_idx"] = edge_sample_index
            result["edge_phase_rad"] = edge_phase_rad

            window_start = result["fft_window_start_idx"]
            window_end = result["fft_window_end_idx"]
            window_signal_before = np.array(signal_iq[window_start:window_end], copy=True)
            window_is_dirty = window_start < dirty_end

            subtraction = self.apply_best_peak_subtraction(window_signal_before, result, window_is_dirty)

            if subtraction is not None:
                synthesized_signal_iq[window_start:window_end] = subtraction.get("subtraction_tone")
                result["subtraction"] = subtraction
                pulse_offset_hz = subtraction["selected_peak"]["frequency_hz"]
            else:
                pulse_offset_hz = 0

            pulses.append(
                f"{self.time_chunk_start + edge_time:.6f},{pulse_offset_hz},"
                f"{plateau_db:.2f},{plateau_db + magnitude_delta_db:.2f},{snr_db:.2f},"
                f"{1e3 * len(window_signal_before) / self.samp_rate:.3f},{len(window_signal_before)},"
                f"2048,0,0,{result['edge_phase_rad']:.6f}"
            )

            if subtraction is not None:
                subtraction.setdefault("iq_before", window_signal_before)
                if subtraction["signal_after"] is not None:
                    signal_iq[window_start:window_end] = subtraction["signal_after"]
                    signal_magnitude[window_start:window_end] = np.abs(subtraction["signal_after"])
                    dirty_end = max(dirty_end, window_end)

            # Zero the detected peak AND the look-ahead slot to prevent re-detection
            # on the same edge (next iteration sees corr=high, prev_corr=0, fires again).
            correlations[i] = 0.0
            if i + 1 < len(correlations):
                correlations[i + 1] = 0.0
            # Block next 2 iterations: prevents raw_xcorr reuse double-detection when
            # apply_best_peak_subtraction returns None and dirty_end is not advanced.
            blocked_until = i + 3

            self.times["subtract_signal"] += time.time() - tmp

        self.prev_chunk_correlations = correlations[-2:]
        return pulses, np.array(correlations, dtype=float), signal_iq, synthesized_signal_iq
        
    def rolling_noise_buffer( self, new_samples ):    
        idx = len(self.noise_buffer) % self.noise_buffer_len
        self.noise_buffer[idx:idx+len(new_samples)] = new_samples
        return np.nanmean(self.noise_buffer)

    def detect_edges(self, signal_iq = None):
        
        signal_magnitude = np.abs(signal_iq)
        template_length = len(self.edge_template)
        signal_length = len(signal_magnitude)
        correlation_length = signal_length - template_length + 1

        if correlation_length <= 0:
            # Not enough data to correlate
            return [], np.array([]), template_length

        correlations = [] if self.prev_chunk_correlations is None else self.prev_chunk_correlations
        pulses = []
        
        for i in range(correlation_length):
            tmp = time.time()
            # Inspect a segment for an edge
            segment = signal_magnitude[i:i + template_length] # We want to add in template_length sample from the previous chunk signal, too
            self.times["get_segment"] += time.time() - tmp
            tmp = time.time()
            # segment_detrended = signal.detrend(segment, type='linear')
            segment_detrended = self.fast_detrend_linear(segment)
            # segment_detrended = segment - np.mean(segment)
            self.times["detrend_segment"] += time.time() - tmp
            tmp = time.time()
            segment_energy = np.linalg.norm(segment_detrended)
            self.times["normalize_segment"] += time.time() - tmp


            # tmp = time.time()
            # segment_norm = segment - np.mean(segment)
            # segment_norm = segment_norm / np.linalg.norm(segment_norm)
            # correlations.append(np.correlate(segment_norm, self.edge_template))
            # self.times["normalize_segment"] += time.time() - tmp

            # Get the segment correlation
            correlation = float(np.dot(segment_detrended, self.edge_template) / segment_energy) if segment_energy > 0 else 0.0
            correlations.append(correlation)

            # Look at the previous correlation
            candidate_idx = len(correlations) - 2
            # Don't look at the first one, though
            if candidate_idx >= 1:
                tmp = time.time()
                pulse, signal_after = self.consider_peak(correlations, candidate_idx, segment, signal_iq)
                self.times["consider_peak"] += time.time() - tmp
                if pulse:
                    pulses.append(pulse)

                    # Before continuing, we want to subtract the best peak from the signal.
                    if signal_after is not None:
                        signal_iq = signal_after
                        signal_magnitude = np.abs(signal_iq)

        if correlations:
            # Store the last two correlations to look at during the next chunk
            self.prev_chunk_correlations = correlations[-2:] 

        return pulses, np.array(correlations, dtype=float), signal_iq

    def consider_peak(self, correlations, idx, segment, signal_iq):

        value = correlations[idx]
        prev_value = correlations[idx - 1]
        next_value = correlations[idx + 1] if idx < len(correlations) - 1 else -np.inf # I don't like this

        # If the correlation is lower than either of the surrounding correlations, return
        if value <= prev_value or value <= next_value:
            # processed_indices.add(idx)
            return None, None

        window_start = max(0, idx - self.edge_window_samples)
        trailing_values = correlations[window_start:idx]

        trailing_max = max(trailing_values)
        # If there are trailing correlations which are stronger, return empty handed
        # Should probably check if the magnitude is strong enough at the same time?
        if value <= trailing_max:   
            # processed_indices.add(idx)
            return None, None

        # Make sure it meets minimum correlation threshold
        if value < self.edge_correlation_threshold:
            # processed_indices.add(idx)
            return None, None

        template_length = len(self.edge_template)

        edge_offset = max(1, template_length // 3)
        baseline = float(np.mean(segment[:edge_offset]))
        plateau = float(np.mean(segment[-edge_offset:]))
        magnitude_delta = plateau - baseline
        # Make sure the magnitude changes by a threshold amount
        if magnitude_delta < self.edge_magnitude_threshold:
            # processed_indices.add(idx)
            return None, None

        peak_strength = value - trailing_max
        edge_index = idx + edge_offset # Includes the ramp up, but not the baseline
        edge_time = edge_index / self.samp_rate

        # If the pulse overlaps with the edge of the chunk, return empty handed
        if edge_index + self.pulse_len_samples > len(signal_iq):
            return None, None

        sample_value = signal_iq[edge_index]
        edge_phase_rad = float(np.arctan2(sample_value.imag, sample_value.real))

        if self.verbose:
            print(
                f"Rising edge detected at {edge_time:.6f} s (correlation: {value:.3f}, "
                f"strength: {peak_strength:.3f}, delta: {magnitude_delta:.6f})"
            )
        tmp = time.time()
        pulse, signal_after = self.handle_rising_edge({
            "edge_time": float(edge_time),
            "correlation_value": float(value),
            "peak_strength": float(peak_strength),
            "magnitude_delta": float(magnitude_delta),
            "correlation_index": int(idx),
            "template_length": int(template_length),
            "edge_idx": int(edge_index),
            "edge_phase_rad": float(edge_phase_rad)
        }, signal_iq)
        self.times["handle_peak"] += time.time() - tmp

        # rising_edge = (edge_time, float(value), float(peak_strength), float(magnitude_delta))
        return pulse, signal_after

    def create_edge_template(self):
        """Create matched filter template for rising edge detection."""
        segment_samples = max(1, int(np.round(self.edge_duration_ms * 1e-3 * self.samp_rate / 3)))
        total_samples = max(3, segment_samples * 3)

        # Start with all zeros
        template = np.zeros(total_samples)
        # Get the middle 1/3 of indices
        ramp_indices = np.arange(segment_samples, 2 * segment_samples) 
        # Ramp up the middle indices
        template[ramp_indices] = np.linspace(0, 1, len(ramp_indices))
        # Make the last 1/3 of samples all 1
        template[2 * segment_samples:] = 1.0
        # Detrend template
        template = template - np.mean(template)
        # Normalize template
        template /= np.linalg.norm(template)
        # Return the detrended, normalized template
        return template

    def handle_rising_edge(self, edge_event, signal_iq):
        tmp = time.time()
        result = self.compute_edge_fft(
            signal_iq,
            edge_event["edge_time"]
        )
        self.times["compute_fft"] += time.time() - tmp
        if result is not None:
            result.update({
                "edge_idx": edge_event["edge_idx"],
                "edge_phase_rad": edge_event["edge_phase_rad"],
                "correlation_value": edge_event["correlation_value"],
                "peak_strength": edge_event["peak_strength"],
                "peak_strength_db": 20.0 * np.log10(edge_event["peak_strength"] + 1e-12),
                "magnitude_delta": edge_event["magnitude_delta"],
                "magnitude_delta_db": 20.0 * np.log10(edge_event["magnitude_delta"] + 1e-12),
            })
            window_iq = np.array( signal_iq[result["fft_window_start_idx"]:result["fft_window_end_idx"]] )
            window_signal_before = np.array( window_iq, copy=True )
            tmp = time.time()
            subtraction = self.apply_best_peak_subtraction( window_iq, result)
            self.times["apply_best_peak"] += time.time() - tmp
            if subtraction is not None:
                tone_samples = subtraction.get("subtraction_tone")
                signal_after = subtraction.get("signal_after")
                if signal_after is None or len(signal_after) != len(window_signal_before):
                    signal_after = window_iq
                
                # print(f"({self.time_chunk_start + edge_event["edge_time"]:.4f}) Signal magnitude before {np.sum(np.abs(window_iq)):.4f} (n={len(window_iq)})")

                signal_iq[result["fft_window_start_idx"]:result["fft_window_end_idx"]] = signal_after
                result["subtraction"] = subtraction
                pulse_offset_hz = subtraction["selected_peak"]["frequency_hz"]
                # print(f"({self.time_chunk_start + edge_event["edge_time"]:.4f}) Signal magnitude after {np.sum(np.abs(signal_after)):.4f} (n={len(signal_after)})")
            else:
                tone_samples = None
                signal_after = window_iq
                pulse_offset_hz = 0

            self.pulses.append(result)

            pulse = f"{self.time_chunk_start + edge_event["edge_time"]:.6f},{pulse_offset_hz},{result["peak_strength_db"]:.2f},{result["peak_strength_db"] + result["magnitude_delta_db"]:.2f},{result["magnitude_delta_db"]:.2f},{1e3*len(window_iq)/self.samp_rate},{len(window_iq)},{2048},{0},{0}"

            return pulse, signal_iq
        
        return None, None
    
    def apply_best_peak_subtraction(self, window_iq, fft_result, window_is_dirty=False):
        """Identify and subtract the peak that most reduces mean magnitude over the FFT window."""
        start = fft_result["fft_window_start_idx"]
        end = fft_result["fft_window_end_idx"]
        window_mag = np.abs(window_iq)

        segment = window_iq.copy()

        original_mean = float(np.mean(np.abs(segment)))
        best_reduction = 0.0
        best_peak = None
        best_tone = None
        best_metadata = None
        # Interference fallback: track the peak with the highest body amplitude / window mean ratio.
        # When two pulses destructively interfere, the window mean is suppressed while the body
        # amplitude (measured in the pre-overlap region) stays at the true pulse amplitude.
        # A ratio significantly above 1.0 indicates interference-induced suppression.
        INTERFERENCE_AMP_RATIO = 2.0
        fallback_amp_ratio = 0.0
        fallback_peak = None
        fallback_tone = None
        fallback_metadata = None

        window_len = len(segment)

        edge_idx = fft_result["edge_idx"]
        edge_phase_hint = fft_result["edge_phase_rad"]
        edge_sample_index = start
        sample_offset = 0

        for peak in fft_result["peaks"]:

            # Refine frequency and phase via phase regression on the early body region.
            # Using only the first 2*ramp_samp samples of the body avoids contamination from
            # overlapping pulses while staying clear of the ramp (where amplitude is 0→A).
            ramp_samp = max(1, self.edge_window_samples // 3)
            body_start = ramp_samp
            # Cap body at 2*ramp_samp to keep clear of a potential overlapping pulse
            body_end = min(body_start + 2 * ramp_samp, window_len - ramp_samp)
            body_end = max(body_end, body_start + 2)
            body_len = body_end - body_start
            t_body = np.arange(body_len, dtype=np.float64)

            # Demodulate to near-DC at the coarse FFT frequency
            coarse_freq = peak["frequency_hz"]
            body_iq = window_iq[body_start:body_end]
            demod = body_iq * np.exp(-1j * 2.0 * np.pi * coarse_freq * t_body / self.samp_rate)

            # Linear regression on unwrapped phase → slope = 2π*Δf/sr, intercept = phase at body_start
            try:
                phases = np.unwrap(np.angle(demod))
                poly = np.polyfit(t_body, phases, 1)
                freq_delta = poly[0] * self.samp_rate / (2.0 * np.pi)
                refined_freq = coarse_freq + freq_delta
                phase_at_body = poly[1]  # phase at body_start (t=0)
                # Amplitude from coherent average of demodulated body
                refined_amp = float(np.abs(np.mean(demod)))
                if refined_amp < 1e-12:
                    refined_amp = peak["amplitude"]
            except Exception:
                refined_freq = coarse_freq
                phase_at_body = float(np.angle(np.mean(demod))) if np.abs(np.mean(demod)) > 1e-15 else 0.0
                refined_amp = peak["amplitude"]

            # Shift phase back to window sample 0
            phase_at_edge = phase_at_body - 2.0 * np.pi * refined_freq * body_start / self.samp_rate

            # Synthesize a tone using the refined frequency, amplitude, and phase
            synthetic = self.synthesize_tone(
                amplitude=refined_amp,
                frequency_hz=refined_freq,
                phase_rad=phase_at_edge,
                length=window_len,
                dtype=window_iq.dtype,
            )
            if len(synthetic) != window_len:
                continue

            envelope = self.get_template_envelope( window_len, self.edge_window_samples )
            # Taper the synthetic signal
            tapered = synthetic * envelope
            # Created a modified segment with the tapered signal removed
            modified = segment - tapered
            # Get the mean magnitude of the segment
            new_mean = float(np.mean(np.abs(modified)))
            # Get the difference in mean magnitude between the original and modified the segments
            reduction = original_mean - new_mean

            if reduction > best_reduction:
                best_reduction = reduction
                best_peak = peak
                best_tone = tapered
                best_metadata = {
                    "phase_start_rad": phase_at_edge,
                }

            # Track interference fallback: body amplitude >> window mean implies destructive interference
            amp_ratio = refined_amp / (original_mean + 1e-12)
            if amp_ratio > fallback_amp_ratio:
                fallback_amp_ratio = amp_ratio
                fallback_peak = peak
                fallback_tone = tapered
                fallback_metadata = {
                    "phase_start_rad": phase_at_edge,
                }

        # Primary path: a peak improved the window mean
        if best_peak is not None:
            pass  # use best_peak / best_tone already set
        # Interference fallback: no peak improved the mean, body amplitude >> suppressed window mean,
        # and the detection is on the original (non-dirty) signal to avoid cascading from bad subtractions.
        elif not window_is_dirty and fallback_amp_ratio > INTERFERENCE_AMP_RATIO and fallback_peak is not None:
            best_peak = fallback_peak
            best_tone = fallback_tone
            best_metadata = fallback_metadata
        else:
            return None

        if best_tone is None or best_metadata is None:
            return None

        window_iq -= best_tone
        
        result = {
            "selected_peak": best_peak,
            "mean_before": original_mean,
            "mean_after": float(np.mean(np.abs(window_iq))),
            "mean_reduction": best_reduction,
            "subtraction_tone": best_tone.copy(),
            "signal_before": segment.copy(),
            "signal_after": window_iq.copy(),
        }
        if best_metadata is not None:
            result.update(best_metadata)
        if edge_sample_index is not None:
            result["edge_sample_index"] = edge_sample_index
            result["sample_offset"] = sample_offset

        return result


        # Fine-tune frequency and phase using least squares optimization
        from scipy.optimize import minimize

        initial_freq = best_peak["frequency_hz"]
        initial_amplitude = best_peak["amplitude"]
        initial_phase = best_metadata["phase_start_rad"]

        def objective(params):
            freq, phase = params
            synthetic = self.synthesize_tone(
                amplitude=initial_amplitude,
                frequency_hz=freq,
                phase_rad=phase,
                length=window_len,
                dtype=window_iq.dtype,
            )
            tapered = synthetic * self.envelope
            residual = segment - tapered
            residual_mean_magnitude = np.mean(np.abs(residual))
            # Return negative because minimize() finds minimum, but we want minimum residual magnitude
            return residual_mean_magnitude

        # Set bounds: allow ±5% frequency variation and full phase range
        freq_bounds = (initial_freq * 0.95, initial_freq * 1.05)
        phase_bounds = (initial_phase - np.pi, initial_phase + np.pi)
        bounds = [freq_bounds, phase_bounds]
        
        result_opt = minimize(
            objective,
            x0=[initial_freq, initial_phase],
            bounds=bounds,
            method='L-BFGS-B'
        )

        if result_opt.success:
            optimized_freq, optimized_phase = result_opt.x
            optimized_synthetic = self.synthesize_tone(
                amplitude=initial_amplitude,
                frequency_hz=optimized_freq,
                phase_rad=optimized_phase,
                length=window_len,
                dtype=window_iq.dtype,
            )
            best_tone = optimized_synthetic * self.envelope
            
            # Update metadata with optimized values
            best_metadata["optimized_freq_hz"] = float(optimized_freq)
            best_metadata["optimized_phase_rad"] = float(optimized_phase)
            best_metadata["freq_shift_hz"] = float(optimized_freq - initial_freq)
            best_metadata["phase_shift_rad"] = float(optimized_phase - initial_phase)
            
            if self.verbose:
                print(f"    Fine-tuned: {initial_freq/1e3:.3f} → {optimized_freq/1e3:.3f} kHz "
                    f"(Δ{(optimized_freq-initial_freq):.1f} Hz), "
                    f"phase {initial_phase:.3f} → {optimized_phase:.3f} rad")

        window_iq_after = window_iq - best_tone

        result = {
            "selected_peak": best_peak,
            "mean_before": original_mean,
            "mean_after": float(np.mean(np.abs(window_iq_after))),
            "mean_reduction": best_reduction,
            "subtraction_tone": best_tone,
            "signal_before": window_iq,
            "signal_after": window_iq_after
        }
        if best_metadata is not None:
            result.update(best_metadata)
        if edge_sample_index is not None:
            result["edge_sample_index"] = edge_sample_index
            result["sample_offset"] = sample_offset
        return result

    def synthesize_tone(self, amplitude, frequency_hz, phase_rad, length, dtype):
        """Create a complex sinusoid with the supplied amplitude, frequency, and phase."""
        if length <= 0:
            return np.zeros(0, dtype=dtype)

        sample_indices = np.arange(length, dtype=np.float64)
        phase = phase_rad + 2.0 * np.pi * frequency_hz * sample_indices / self.samp_rate
        tone = amplitude * np.exp(1j * phase)
        return tone.astype(dtype, copy=False)

    def compute_edge_fft(self, signal_iq, edge_time_s):
        """Compute a short FFT snapshot following a rising edge detection."""
        # Get start and end sample index for pulse
        start_idx = int(round(edge_time_s * self.samp_rate))
        end_idx = start_idx + self.pulse_len_samples

        if start_idx < 0 and self.verbose:
            # This isn't supposed to happen...
            if self.verbose:
                print(f" Start sample index is negative???")
        elif end_idx > len(signal_iq):
            if self.prev_chunk_last_pulse_samples is None:
                self.prev_chunk_last_pulse_samples = signal_iq[start_idx:len(signal_iq)-1]
            if self.verbose:
                print("Pulse end sample index runs past the end of chunk")
            return None


        # Samples for pulse window
        window_iq = signal_iq[start_idx:end_idx]
        window_len = end_idx - start_idx
        # Ensure FFT is a power of 2
        fft_length = 8 * max(8, 1 << int(np.ceil(np.log2(window_len)))) 
        # Get the FFT
        spectrum = np.fft.fft(window_iq, n=fft_length)
        freq_axis = np.fft.fftfreq(fft_length, d=1 / self.samp_rate)
        fft_mag = np.abs(spectrum)
        fft_mag_norm = fft_mag / window_len
        
        # Get the peak magnitude
        peak_indices, _ = signal.find_peaks(fft_mag)
        if peak_indices.size == 0:
            peak_indices = np.array([int(np.argmax(fft_mag))])

        max_idx = int(peak_indices[np.argmax(fft_mag[peak_indices])])
        amplitude_threshold = fft_mag[max_idx] / (self.fft_amplitude_threshold_db ** 0.5)

        selected_indices = [
            int(idx) for idx in peak_indices
            if fft_mag[idx] >= amplitude_threshold
        ]

        selected_indices.sort(key=lambda idx: fft_mag[idx], reverse=True)


        dominant_idx = selected_indices[0]
        dominant_freq_hz = float(freq_axis[dominant_idx])
        dominant_amplitude = float(fft_mag_norm[dominant_idx])
        dominant_power_db = 20.0 * np.log10(dominant_amplitude + 1e-12)
        dominant_phase_rad = float(np.angle(spectrum[dominant_idx]))

        shifted_spectrum = np.fft.fftshift(spectrum)
        shifted_freq = np.fft.fftshift(freq_axis)
        shifted_mag_db = 20.0 * np.log10(np.abs(shifted_spectrum) / window_len + 1e-12)

        shifted_marker_x = []
        shifted_marker_y = []
        peaks = []
        for idx in selected_indices:
            amp = float(fft_mag_norm[idx])
            phase = float(np.angle(spectrum[idx]))
            power_db = 20.0 * np.log10(amp + 1e-12)
            freq_hz = float(freq_axis[idx])

            shifted_idx = int((idx + fft_length // 2) % fft_length)
            shifted_marker_x.append(shifted_freq[shifted_idx] / 1e3)
            shifted_marker_y.append(shifted_mag_db[shifted_idx])

            peaks.append({
                "frequency_hz": freq_hz,
                "amplitude": amp,
                "power_db": power_db,
                "phase_rad": phase,
            })
            if self.verbose:
                print(
                    f"Found peak at {freq_hz} hz, {power_db} dB"
                )


        # fig.write_html(file_name)
        if self.verbose:
            print(
                f"({len(peaks)} peaks within {self.fft_amplitude_threshold_db:.1f} dB of max)"
            )

        return {
            "edge_time_s": edge_time_s,
            "dominant_freq_hz": dominant_freq_hz,
            "dominant_amplitude": dominant_amplitude,
            "dominant_power_db": dominant_power_db,
            "dominant_phase_rad": dominant_phase_rad,
            "fft_window_start_idx": start_idx,
            "fft_window_end_idx": end_idx,
            "peaks": peaks,
            "threshold_db": self.fft_amplitude_threshold_db,
        }


    def estimate_phase_at_sample(self, signal_data, frequency_hz, sample_index,
                                neighborhood=64):
        """Estimate the component phase for ``frequency_hz`` at ``sample_index``."""
        if sample_index < 0 or sample_index >= len(signal_data):
            return None

        if neighborhood <= 1:
            value = signal_data[sample_index]
            return float(np.arctan2(value.imag, value.real))

        half = max(1, neighborhood // 2)
        start = max(0, sample_index - half)
        end = min(len(signal_data), sample_index + half)
        if end - start <= 1:
            value = signal_data[sample_index]
            return float(np.arctan2(value.imag, value.real))

        relative_indices = np.arange(start, end, dtype=np.float64) - sample_index
        segment = signal_data[start:end]
        reference = np.exp(-1j * 2.0 * np.pi * frequency_hz * relative_indices / self.samp_rate)
        coeff = np.dot(segment, reference)
        if np.abs(coeff) < 1e-15:
            value = signal_data[sample_index]
            return float(np.arctan2(value.imag, value.real))
        return float(np.arctan2(coeff.imag, coeff.real))

    def get_template_envelope(self, window_len, edge_len):
        # Ramp up only — no ramp-down at the end. A taper-down creates a rising
        # residual at the tail of the subtracted window that the detector mistakes
        # for a real rising edge, causing systematic false positives ~2.4 ms after
        # every true detection.
        ramp_samples = max(1, min(edge_len // 3, window_len))
        envelope = np.ones(window_len, dtype=np.float64)
        envelope[:ramp_samples] = np.linspace(0.0, 1.0, ramp_samples, endpoint=False)
        if ramp_samples > 1:
            envelope[ramp_samples - 1] = 1.0
        return envelope

    def fast_detrend_linear(self, x):
        n = len(x)
        t = np.arange(n, dtype=np.float32)
        # Fit line: y = a*t + b
        t_mean = np.mean(t)
        x_mean = np.mean(x)
        cov = np.dot(t - t_mean, x - x_mean)
        var = np.dot(t - t_mean, t - t_mean)
        a = cov / var
        b = x_mean - a * t_mean
        return x - (a * t + b)

    def __del__(self):
        if hasattr(self, 'csvfile'):
            
            try:
                self.csvfile.close()
            except:
                pass