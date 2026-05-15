import csv
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
import time
import os
import sys
sys.path.insert(0, os.path.dirname(__file__) or ".")
import detect_pulse_2 as dp2

# Configuration
# path = "2025_10_20_raw_iq_airspyhf_192khz_superimposed_bursts.bin"
# path = "baseband_166376000Hz_17-03-43_04-11-2025.raw"
# path = "raw_iq_airspyhf_192000_4000.bin"
path = "test_signal.iq"
sample_rate = 192e3  # Hz
# burst 1: 4.5-5.2 s
# burst 2: 15.5-16.2 s
t_start = 0
t_stop = 1
s_start = int(t_start * sample_rate)
s_stop = int(t_stop * sample_rate)

# Filtering & decimation settings
decimation_factor = 4
passband_edge_hz = 6e3
stopband_edge_hz = 12e3
filter_attenuation_db = 60
correlation_threshold = 0.98
pre_fft_results = 0

# Matched filter configuration
edge_template_duration_ms = 0.3  # Template span and correlation lookback
edge_fft_offset_ms = 2.55
magnitude_threshold_db = -40 # dB
min_snr_db = 6 # dB
# magnitude_threshold_db = 2e-4 # dB

enable_edge_plots = False  # Toggle FFT and time-domain plot generation

true_pulse_catalog_path = "test_signal.csv"

def create_edge_template(sample_rate):
    """Create matched filter template for rising edge detection."""
    segment_samples = max(1, int(np.round(edge_template_duration_ms * 1e-3 * sample_rate / 3)))
    total_samples = max(3, segment_samples * 3)

    template = np.zeros(total_samples)
    ramp_indices = np.arange(segment_samples, 2 * segment_samples)
    template[ramp_indices] = np.linspace(0, 1, len(ramp_indices))
    template[2 * segment_samples:] = 1.0
    template = template - np.mean(template)
    template = template / np.linalg.norm(template)
    print("TEMPLATE LENGTH:",len(template))
    return template


def load_pulse_catalog(path):
    """Load pulse metadata from test_signal.csv (start_idx, end_idx, freq, phase, mags)."""
    pulses = []
    if not path:
        return pulses
    try:
        with open(path, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                try:
                    start_time_seconds = int(row["start_idx"]) / sample_rate
                    frequency_hz = float(row["freq"])
                    amplitude = float(row["mags"])
                    phase_radians = float(row["phase"])
                except (KeyError, ValueError, TypeError):
                    continue
                pulses.append((i + 1, start_time_seconds, frequency_hz, amplitude, phase_radians))
    except FileNotFoundError:
        print(f"Reference pulse catalog not found: {path}")
    except OSError as exc:
        print(f"Failed to load reference pulse catalog '{path}': {exc}")
    return pulses


def detect_edges_matched_filter(iq, sr, template):
    """Detect rising edges using detect_pulse_2.Pulse_Detector.

    Returns (rising_edges, correlations, mag_deltas).
    rising_edges items: (time_s, corr_score, plateau_db, magnitude_delta_db)
    mag_deltas: per-sample (plateau - baseline) array aligned with correlations.
    """
    global pre_fft_results

    # Capture pre-subtraction magnitude for mag_deltas computation
    sig_mag_orig = np.abs(iq)

    det = dp2.Pulse_Detector(output_type="test", samp_rate=sr,
                             verbose=False, pulse_len_ms=2.5)
    output_lines, correlations, signal_after, _synthesized = det.detect([iq.copy()])

    rising_edges = []
    for line in output_lines:
        parts = line.split(",")
        if len(parts) < 5:
            continue
        ts        = float(parts[0])
        plateau_db = float(parts[2])
        peak_db    = float(parts[3])
        mag_delta_db = peak_db - plateau_db

        pre_fft_results += 1
        rising_edges.append((ts, 0.98, plateau_db, mag_delta_db))

        result = compute_edge_fft(iq, sr, ts, len(rising_edges))
        if result is not None:
            edge_fft_results.append(result)

    # Update iq in-place to the post-subtraction signal for visualization
    pulse_samples = det.pulse_len_samples
    residual = np.concatenate([signal_after, iq[-pulse_samples:]]).astype(np.complex64)
    iq[:len(residual)] = residual

    # Per-sample plateau-baseline magnitude delta using O(1) prefix sums
    tmpl_len = len(det.edge_template)
    third = max(1, tmpl_len // 3)
    n_corr = max(0, len(sig_mag_orig) - tmpl_len + 1)
    if n_corr > 0:
        pfx = np.zeros(len(sig_mag_orig) + 1)
        np.cumsum(sig_mag_orig, out=pfx[1:])
        baseline = (pfx[third:third + n_corr] - pfx[:n_corr]) / third
        plateau  = (pfx[tmpl_len:tmpl_len + n_corr] - pfx[2 * third:2 * third + n_corr]) / third
        mag_deltas = plateau - baseline
    else:
        mag_deltas = np.zeros(0)

    return rising_edges, correlations, mag_deltas


def plot_edge_template(template, sample_rate):
    """Plot the rising-edge matched filter template for visualization."""
    time_axis = np.arange(len(template)) / sample_rate

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis, y=template,
        name='Rising Edge Template',
        line=dict(color='orange', width=2)
    ))

    fig.update_layout(
        title='Matched Filter Template for Edge Detection',
        xaxis_title='Time (s)',
        yaxis_title='Normalized Amplitude',
        template='plotly_white',
        showlegend=True
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    fig.write_html("edge_templates.html")
    print("Edge template visualization saved to 'edge_templates.html'")

def plot_time_domain(iq_data, sample_rate, rising_edges=None,
                        rising_corr=None, template_length=None,
                        original_iq_data=None, true_rising_edges=None,
                        edge_fft_results=None, mag_deltas=None):
    """Analyze time domain characteristics and overlay detected and reference edges."""
    time_axis = (t_start + np.arange(len(iq_data)) / sample_rate) * 1000

    original_magnitude = None
    original_time_axis = None
    original_i = None
    original_q = None
    original_phase = None
    if original_iq_data is not None:
        if len(original_iq_data) == len(iq_data):
            original_magnitude = np.abs(original_iq_data)
            original_time_axis = time_axis
            original_i = np.real(original_iq_data)
            original_q = np.imag(original_iq_data)
            original_phase = np.angle(original_iq_data)
        else:
            min_len = min(len(iq_data), len(original_iq_data))
            if min_len > 0:
                original_magnitude = np.abs(original_iq_data[:min_len])
                original_time_axis = time_axis[:min_len]
                original_i = np.real(original_iq_data[:min_len])
                original_q = np.imag(original_iq_data[:min_len])
                original_phase = np.angle(original_iq_data[:min_len])

    has_corr = rising_corr is not None and len(rising_corr) > 0

    subplot_titles = ['I/Q Time Domain', 'Signal Magnitude']
    corr_row = None
    if has_corr:
        corr_row = len(subplot_titles) + 1
        subplot_titles.append('Matched Filter Correlations')
    mag_deltas_row = len(subplot_titles) + 1
    subplot_titles.append('Magnitude Delta')
    phase_row = len(subplot_titles) + 1
    subplot_titles.append('Signal Phase')
    total_rows = len(subplot_titles)

    # Create subplots with Plotly
    fig = make_subplots(
        rows=total_rows, cols=1,
        subplot_titles=tuple(subplot_titles),
        vertical_spacing=0.05, shared_xaxes=True
    )
    
    # Plot I and Q components
    if original_time_axis is not None and original_i is not None and original_q is not None:
        fig.add_trace(
            go.Scatter(x=original_time_axis, y=original_i,
                      name='I (original)', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=original_time_axis, y=original_q,
                      name='Q (original)', line=dict(color='red')),
            row=1, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(x=time_axis, y=np.real(iq_data),
                      name='I (after subtraction)', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_axis, y=np.imag(iq_data),
                      name='Q (after subtraction)', line=dict(color='red')),
            row=1, col=1
        )
    
    # Plot magnitude
    if original_magnitude is not None and original_time_axis is not None:
        fig.add_trace(
            go.Scatter(x=original_time_axis, y=original_magnitude,
                       name='Magnitude (original)', line=dict(color='gray')),
            row=2, col=1
        )

    fig.add_trace(
        go.Scatter(x=time_axis, y=np.abs(iq_data),
                  name='Magnitude (after subtraction)', line=dict(color='green')),
        row=2, col=1
    )
    
    base_time_axis_sec = np.arange(len(iq_data)) / sample_rate if len(iq_data) > 0 else np.array([0.0])
    base_signal = np.abs(iq_data)
    if len(base_signal) == 0:
        base_signal = np.array([0.0])
        base_time_axis_sec = np.array([0.0])

    # Add detected edges to magnitude plot if provided
    if rising_edges:
        rising_times = [(t_start + edge[0]) * 1000 for edge in rising_edges]
        rising_values = [
            np.interp(edge[0], base_time_axis_sec, base_signal, left=base_signal[0], right=base_signal[-1])
            for edge in rising_edges
        ]
        fig.add_trace(
            go.Scatter(x=rising_times, y=rising_values,
                      mode='markers', marker=dict(color='red', size=8, symbol='triangle-up'),
                      name='Detected Rising Edges', showlegend=True),
            row=2, col=1
        )

    # Overlay reference pulses from catalog if provided
    if true_rising_edges:
        true_times = [(edge[1]) * 1000 for edge in true_rising_edges]
        true_values = [
            np.interp(edge[1], base_time_axis_sec, base_signal, left=base_signal[0], right=base_signal[-1])
            for edge in true_rising_edges
        ]
        fig.add_trace(
            go.Scatter(
                x=true_times,
                y=true_values,
                mode='markers',
                marker=dict(color='black', size=8, symbol='triangle-up-open'),
                name='Reference Rising Edges',
                showlegend=True,
            ),
            row=2,
            col=1,
        )
    
    # Plot phase
    if original_time_axis is not None and original_phase is not None:
        fig.add_trace(
            go.Scatter(x=original_time_axis, y=original_phase,
                      name='Phase (original)', line=dict(color='teal')),
            row=phase_row, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(x=time_axis, y=np.angle(iq_data),
                      name='Phase (after subtraction)', line=dict(color='teal')),
            row=phase_row, col=1
        )

    if has_corr:
        rising_offset = 0.0
        if template_length is not None:
            rising_offset = template_length * (1.0 / 3.0)

        if rising_corr is not None and len(rising_corr) > 0:
            rising_time_axis = (t_start + (np.arange(len(rising_corr)) + rising_offset) / sample_rate) * 1000
            fig.add_trace(
                go.Scatter(x=rising_time_axis, y=rising_corr,
                           name='Rising Correlator', line=dict(color='orange')),
                row=corr_row, col=1
            )
    
    # Plot plateau magnitude and ratio for all correlation points
    if mag_deltas is not None and len(mag_deltas) > 0:
        plateau_offset = 0.0
        if template_length is not None:
            plateau_offset = template_length * (1.0 / 3.0)
        
        plateau_time_axis = (t_start + (np.arange(len(mag_deltas)) + plateau_offset) / sample_rate) * 1000
        
        # Plot continuous plateau magnitude
        fig.add_trace(
            go.Scatter(x=plateau_time_axis, y=mag_deltas,
                      mode='lines', line=dict(color='darkblue', width=1),
                      name='Magnitude delta', showlegend=True),
            row=mag_deltas_row, col=1
        )
        
        # Add threshold line for magnitude (convert threshold from dB to normal units)
        if magnitude_threshold_db is not None:
            threshold_normal = 10.0 ** (magnitude_threshold_db / 10.0)
            fig.add_hline(y=threshold_normal, line=dict(color='red', dash='dash', width=1),
                         annotation_text=f'Threshold: {magnitude_threshold_db} dB',
                         annotation_position='right', row=mag_deltas_row, col=1)
        
        # Mark detected edges on plateau magnitude plot
        if edge_fft_results:
            edge_times_ms = [result['edge_time_ms'] for result in edge_fft_results]
            # Convert edge plateau values from dB to normal units for display
            edge_plateau_normal = [10.0 ** (result.get('dominant_power_db', -120) / 10.0) for result in edge_fft_results]
            fig.add_trace(
                go.Scatter(x=edge_times_ms, y=edge_plateau_normal,
                          mode='markers', marker=dict(color='red', size=8, symbol='triangle-up'),
                          name='Detected Edges', showlegend=True),
                row=mag_deltas_row, col=1
            )
    
    # Update layout
    fig.update_layout(
        title_text="Time Domain Analysis with Edge Detection",
        template='plotly_white',
        height=200 * total_rows
    )
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Time (ms)", row=phase_row, col=1)
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)
    if has_corr and corr_row is not None:
        fig.update_yaxes(title_text="Correlation Score", row=corr_row, col=1)
    fig.update_yaxes(title_text="Magnitude Delta", row=mag_deltas_row, col=1)
    fig.update_yaxes(title_text="Phase (radians)", row=phase_row, col=1)
    
    # Add grid to all subplots
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    fig.write_html("time_domain_analysis.html")


def summarize_pulse_collection(label, pulses):
    print(f"\n{label}: {len(pulses)}")
    if not pulses:
        return

    eps = 1e-12
    times_ms = [float(p[1]) * 1000.0 for p in pulses]
    freqs_khz = [float(p[2]) / 1000.0 for p in pulses]
    amplitudes_db = [10.0 * np.log10(max(abs(float(p[3])), eps)) for p in pulses]

    def summarize(values):
        mean_val = float(np.mean(values))
        if len(values) > 1:
            std_val = float(np.std(values, ddof=1))
        else:
            std_val = 0.0
        return mean_val, std_val

    time_mean, time_std = summarize(times_ms)
    freq_mean, freq_std = summarize(freqs_khz)
    amp_mean, amp_std = summarize(amplitudes_db)

    print(f"  Start time (ms): mean {time_mean:.3f}, std {time_std:.3f}")
    print(f"  Frequency (kHz): mean {freq_mean:.3f}, std {freq_std:.3f}")
    print(f"  Amplitude (dB): mean {amp_mean:.3f}, std {amp_std:.3f}")


def plot_timing_error_analysis(rising_edges, reference_pulses, match_tol_s=1e-3):
    """Overlay detected vs reference times and visualise timing error distribution."""
    if not rising_edges or not reference_pulses:
        print("Skipping timing error analysis: no data.")
        return

    ref_times_s = sorted(p[1] for p in reference_pulses)
    det_times_s = sorted(edge[0] for edge in rising_edges)

    def greedy_match(det_list, ref_list, tol):
        used = set()
        pairs = []
        for dt in det_list:
            best_j, best_err = None, tol
            for j, rt in enumerate(ref_list):
                if j in used:
                    continue
                err = abs(dt - rt)
                if err < best_err:
                    best_err = err
                    best_j = j
            if best_j is not None:
                used.add(best_j)
                pairs.append((dt, ref_list[best_j]))
        return pairs

    pairs = greedy_match(det_times_s, ref_times_s, match_tol_s)
    if not pairs:
        print("No matched detections for timing error analysis.")
        return

    errors_us = np.array([(dt - rt) * 1e6 for dt, rt in pairs])
    ref_ms    = np.array([rt * 1000.0 for _, rt in pairs])

    mean_us = float(np.mean(errors_us))
    std_us  = float(np.std(errors_us))
    n       = len(errors_us)

    # Histogram + Gaussian fit
    n_bins = max(10, n // 4)
    counts, bin_edges = np.histogram(errors_us, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_width   = float(bin_edges[1] - bin_edges[0])

    x_fit = np.linspace(errors_us.min() - 2 * std_us, errors_us.max() + 2 * std_us, 400)
    safe_std = max(std_us, 1e-6)
    gauss = (n * bin_width / (safe_std * np.sqrt(2 * np.pi))
             * np.exp(-0.5 * ((x_fit - mean_us) / safe_std) ** 2))

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Timing Error Histogram  (n={n},  mean={mean_us:.1f} us,  1s={std_us:.1f} us)',
            'Timing Error vs Reference Pulse Time',
        ),
        vertical_spacing=0.12,
    )

    fig.add_trace(go.Bar(x=bin_centers, y=counts, name='count',
                         marker_color='steelblue', opacity=0.75), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_fit, y=gauss, name='Gaussian fit',
                             line=dict(color='red', width=2)), row=1, col=1)

    y_max = float(counts.max()) * 1.15 if len(counts) else 1.0
    for xv, color, label in [
        (mean_us,           'darkorange', f'mean {mean_us:.1f} us'),
        (mean_us - std_us,  'green',      f'-1s {mean_us - std_us:.1f} us'),
        (mean_us + std_us,  'green',      f'+1s {mean_us + std_us:.1f} us'),
    ]:
        fig.add_trace(go.Scatter(x=[xv, xv], y=[0, y_max], mode='lines',
                                 line=dict(color=color, dash='dash', width=1.5),
                                 name=label, showlegend=True), row=1, col=1)

    fig.add_trace(go.Scatter(x=ref_ms, y=errors_us, mode='markers',
                             marker=dict(color='steelblue', size=6),
                             name='error per detection', showlegend=True), row=2, col=1)
    fig.add_trace(go.Scatter(x=[ref_ms.min(), ref_ms.max()],
                             y=[mean_us, mean_us], mode='lines',
                             line=dict(color='darkorange', dash='dash', width=1.5),
                             name=f'mean {mean_us:.1f} us', showlegend=False), row=2, col=1)
    fig.add_hrect(y0=mean_us - std_us, y1=mean_us + std_us,
                  fillcolor='green', opacity=0.1,
                  annotation_text=f'±1s = {std_us:.1f} us',
                  annotation_position='top right', row=2, col=1)

    fig.update_layout(title_text='Detection Timing Error Analysis',
                      template='plotly_white', height=600)
    fig.update_xaxes(title_text='Timing error (us)', row=1, col=1, showgrid=True)
    fig.update_xaxes(title_text='Reference pulse time (ms)', row=2, col=1, showgrid=True)
    fig.update_yaxes(title_text='Count', row=1, col=1, showgrid=True)
    fig.update_yaxes(title_text='Error (us)', row=2, col=1, showgrid=True)

    fig.write_html('timing_error_analysis.html')
    print(f"\nTiming error: n={n}, mean={mean_us:.1f} us, 1s={std_us:.1f} us")
    print("Saved 'timing_error_analysis.html'")


def compute_edge_fft(iq_data, sample_rate, edge_time_s, edge_num):
    """Compute a short FFT snapshot following a rising edge detection."""
    offset_samples = max(1, int(round(edge_fft_offset_ms * 1e-3 * sample_rate)))
    window_ms = edge_fft_offset_ms
    window_samples = max(1, int(round(window_ms * 1e-3 * sample_rate)))

    end_sample = int(round(edge_time_s * sample_rate)) + offset_samples
    start_sample = end_sample - window_samples

    if start_sample < 0 or end_sample > len(iq_data):
        print(
            f"Skipping rising edge {edge_num}: insufficient data for {window_ms:.3f} ms FFT window"
        )
        return None

    window = iq_data[start_sample:end_sample]
    if len(window) == 0:
        return None

    windowed = window
    fft_length = 8 * max(8, 1 << int(np.ceil(np.log2(len(windowed)))))
    spectrum = np.fft.fft(windowed, n=fft_length)
    freq_axis = np.fft.fftfreq(fft_length, d=1 / sample_rate)
    magnitude = np.abs(spectrum)
    norm_factor = max(len(windowed), 1)
    magnitude_norm = magnitude / norm_factor

    peak_indices, _ = signal.find_peaks(magnitude)
    if peak_indices.size == 0:
        peak_indices = np.array([int(np.argmax(magnitude))])

    max_idx = int(peak_indices[np.argmax(magnitude[peak_indices])])
    max_amplitude = magnitude[max_idx]
    amplitude_threshold = max_amplitude / 10.0

    selected_indices = [int(idx) for idx in peak_indices if magnitude[idx] >= amplitude_threshold]
    if max_idx not in selected_indices:
        selected_indices.append(max_idx)

    selected_indices = sorted(selected_indices, key=lambda idx: magnitude[idx], reverse=True)
    selected_indices = selected_indices[:3]
    dominant_idx = selected_indices[0]
    dominant_freq_hz = float(freq_axis[dominant_idx])
    dominant_amplitude = float(magnitude_norm[dominant_idx])
    dominant_power_db = 10.0 * np.log10(dominant_amplitude + 1e-12)
    dominant_phase_rad = float(np.angle(spectrum[dominant_idx]))

    shifted_spectrum = np.fft.fftshift(spectrum)
    shifted_freq = np.fft.fftshift(freq_axis)
    shifted_mag_db = 10.0 * np.log10(np.abs(shifted_spectrum) / norm_factor + 1e-12)

    shifted_marker_x = []
    shifted_marker_y = []
    peak_entries = []
    for idx in selected_indices:
        amp = float(magnitude_norm[idx])
        phase = float(np.angle(spectrum[idx]))
        power_db = 10.0 * np.log10(amp + 1e-12)
        freq_hz = float(freq_axis[idx])

        shifted_idx = int((idx + fft_length // 2) % fft_length)
        shifted_marker_x.append(shifted_freq[shifted_idx] / 1e3)
        shifted_marker_y.append(shifted_mag_db[shifted_idx])

        peak_entries.append({
            "frequency_hz": freq_hz,
            "amplitude": amp,
            "power_db": power_db,
            "phase_rad": phase,
        })

    threshold_db = 10.0

    edge_time_ms = (t_start + edge_time_s) * 1000.0
    file_name = None
    if enable_edge_plots:
        file_name = f"edge_fft_{edge_num:03.0f}.html"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=shifted_freq / 1e3,
            y=shifted_mag_db,
            mode='lines',
            name='FFT Magnitude'
        ))
        fig.add_trace(go.Scatter(
            x=shifted_marker_x,
            y=shifted_marker_y,
            mode='markers',
            marker=dict(color='red', size=8),
            name='Peaks (<=10 dB from max)'
        ))

        fig.update_layout(
            title=(
                f'{window_ms:.1f} ms FFT after Rising Edge {edge_num} '
                f'(edge at {edge_time_ms:.3f} ms)'
            ),
            xaxis_title='Frequency (kHz)',
            yaxis_title='Magnitude (dB)',
            template='plotly_white'
        )

        fig.write_html(os.path.join("edge_plots", file_name))
        print(
            f"Saved FFT plot for rising edge {edge_num} to '{file_name}' "
            f"({len(peak_entries)} peaks within {threshold_db:.1f} dB of max)"
        )

    return {
        "edge_index": edge_num,
        "edge_time_ms": edge_time_ms,
        "dominant_freq_hz": dominant_freq_hz,
        "dominant_amplitude": dominant_amplitude,
        "dominant_power_db": dominant_power_db,
        "dominant_phase_rad": dominant_phase_rad,
        "fft_window_start_sample": start_sample,
        "fft_window_end_sample": end_sample,
        "fft_file": file_name,
        "peaks": peak_entries,
        "threshold_db": threshold_db,
    }


def synthesize_tone(amplitude, frequency_hz, phase_rad, length, sample_rate, dtype):
    """Create a complex sinusoid with the supplied amplitude, frequency, and phase."""
    if length <= 0:
        return np.zeros(0, dtype=dtype)

    sample_indices = np.arange(length, dtype=np.float64)
    phase = phase_rad + 2.0 * np.pi * frequency_hz * sample_indices / sample_rate
    tone = amplitude * np.exp(1j * phase)
    return tone.astype(dtype, copy=False)


def estimate_phase_at_sample(signal_data, sample_rate, frequency_hz, sample_index,
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
    reference = np.exp(-1j * 2.0 * np.pi * frequency_hz * relative_indices / sample_rate)
    coeff = np.dot(segment, reference)
    if np.abs(coeff) < 1e-15:
        value = signal_data[sample_index]
        return float(np.arctan2(value.imag, value.real))
    return float(np.arctan2(coeff.imag, coeff.real))


def compute_phase_series(values):
    if len(values) == 0:
        return np.zeros(0, dtype=float)
    raw_phase = np.arctan2(values.imag, values.real)
    return np.mod(raw_phase + 2.0 * np.pi, 2.0 * np.pi)


def plot_edge_time_domain_window(edge_num, edge_time_ms, sample_rate, start_sample,
                                 iq_before, iq_after, subtraction_tone):
    if not enable_edge_plots:
        return None

    window_length = len(iq_before)
    if window_length == 0:
        return None

    if subtraction_tone is None or len(subtraction_tone) != window_length:
        subtraction_tone = np.zeros(window_length, dtype=iq_before.dtype)

    if iq_after is None or len(iq_after) != window_length:
        iq_after = np.zeros(window_length, dtype=iq_before.dtype)

    sample_indices = start_sample + np.arange(window_length)
    time_axis_ms = (t_start + sample_indices / sample_rate) * 1000.0
    window_ms = window_length / sample_rate * 1000.0

    signal_amp = iq_before.real
    tone_amp = subtraction_tone.real
    residual_amp = iq_after.real

    signal_mag = np.abs(iq_before)
    tone_mag = np.abs(subtraction_tone)
    residual_mag = np.abs(iq_after)

    signal_phase = compute_phase_series(iq_before)
    tone_phase = compute_phase_series(subtraction_tone)
    residual_phase = compute_phase_series(iq_after)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=(
                            "Amplitude (Real Component)",
                            "Magnitude",
                            "Phase (radians)"
                        ))

    fig.add_trace(go.Scatter(x=time_axis_ms, y=signal_amp,
                             name='Signal (amplitude)', line=dict(color='blue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=time_axis_ms, y=tone_amp,
                             name='Subtraction Tone (amplitude)', line=dict(color='red')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=time_axis_ms, y=residual_amp,
                             name='Signal After Subtraction (amplitude)', line=dict(color='green')),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=time_axis_ms, y=signal_mag,
                             name='Signal (magnitude)', line=dict(color='blue'), showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=time_axis_ms, y=tone_mag,
                             name='Subtraction Tone (magnitude)', line=dict(color='red'), showlegend=False),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=time_axis_ms, y=residual_mag,
                             name='Signal After Subtraction (magnitude)', line=dict(color='green'), showlegend=False),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=time_axis_ms, y=signal_phase,
                             name='Signal (phase)', line=dict(color='blue'), showlegend=False),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=time_axis_ms, y=tone_phase,
                             name='Subtraction Tone (phase)', line=dict(color='red'), showlegend=False),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=time_axis_ms, y=residual_phase,
                             name='Signal After Subtraction (phase)', line=dict(color='green'), showlegend=False),
                  row=3, col=1)

    fig.update_yaxes(title_text='Amplitude', row=1, col=1)
    fig.update_yaxes(title_text='Magnitude', row=2, col=1)
    fig.update_yaxes(title_text='Phase (rad, mod 2π)', row=3, col=1)
    fig.update_xaxes(title_text='Time (ms)', row=3, col=1)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    fig.update_layout(
        title=(
            f'{window_ms:.2f} ms Window around Rising Edge {edge_num} '
            f'(edge at {edge_time_ms:.3f} ms)'
        ),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0)
    )

    file_name = f"edge_time_domain_{edge_num:03.0f}.html"
    fig.write_html(os.path.join("edge_plots", file_name))
    print(f"Saved time-domain window for rising edge {edge_num} to '{file_name}'")
    return file_name


def apply_best_peak_subtraction(iq, magnitude, sample_rate, fft_result):
    """Identify and subtract the peak that most reduces mean magnitude over the FFT window."""
    start = fft_result["fft_window_start_sample"]
    end = fft_result["fft_window_end_sample"]
    window_length = end - start
    if window_length <= 0 or start < 0 or end > len(iq):
        return None

    segment = iq[start:end].copy()
    if len(segment) == 0:
        return None
    
    # TODO: change to RSS
    original_mean = float(np.mean(np.abs(segment)*np.linspace(2,0,window_length)))
    best_reduction = 0.0
    best_peak = None
    best_tone = None
    best_metadata = None

    window_length = len(segment)
    envelope = np.ones(window_length, dtype=np.float64)
    # Use the correct decimated sample rate for envelope calculation
    ramp_samples = max(1, int(round(edge_template_duration_ms * 1e-3 * sample_rate / 3)))
    ramp_samples = min(ramp_samples, window_length)
    if ramp_samples > 0:
        ramp_up = np.ones(window_length, dtype=np.float64)
        ramp_up[:ramp_samples] = np.linspace(0.0, 1.0, ramp_samples, endpoint=False, dtype=np.float64)
        if ramp_samples < window_length:
            ramp_up[ramp_samples:] = 1.0
        if ramp_samples > 1:
            ramp_up[ramp_samples - 1] = 1.0

        ramp_down = np.ones(window_length, dtype=np.float64)
        ramp_down[-ramp_samples:] = np.linspace(1.0, 0.0, ramp_samples, endpoint=False, dtype=np.float64)
        if ramp_samples < window_length:
            ramp_down[:-ramp_samples] = 1.0
        ramp_down[-1] = 0.0

        envelope = ramp_up * ramp_down
        envelope = envelope.astype(np.float64, copy=False)

    edge_sample_index = fft_result.get("edge_sample_index")
    edge_phase_hint = fft_result.get("edge_phase_rad")
    if edge_sample_index is not None and edge_sample_index < 0:
        edge_sample_index = None
    sample_offset = 0
    if edge_sample_index is not None:
        sample_offset = start - edge_sample_index

    for peak in fft_result["peaks"]:
        phase_at_edge = None
        if edge_sample_index is not None:
            phase_at_edge = estimate_phase_at_sample(
                iq,
                sample_rate,
                peak["frequency_hz"],
                edge_sample_index,
            )
        if phase_at_edge is None and edge_phase_hint is not None:
            phase_at_edge = edge_phase_hint
        if phase_at_edge is None:
            phase_at_edge = peak["phase_rad"]

        if edge_sample_index is not None:
            phase_start = phase_at_edge + 2.0 * np.pi * peak["frequency_hz"] * sample_offset / sample_rate
        else:
            phase_start = phase_at_edge

        synthetic = synthesize_tone(
            amplitude=peak["amplitude"],
            frequency_hz=peak["frequency_hz"],
            phase_rad=phase_start,
            length=window_length,
            sample_rate=sample_rate,
            dtype=iq.dtype,
        )
        if len(synthetic) != window_length:
            continue

        tapered = synthetic * envelope
        modified = segment - tapered
        new_mean = float(np.mean(np.abs(modified)*np.linspace(2,0,window_length)))
        reduction = original_mean - new_mean

        if reduction > best_reduction:
            best_reduction = reduction
            best_peak = peak
            best_tone = tapered
            best_metadata = {
                "phase_at_edge_rad": phase_at_edge,
                "phase_start_rad": phase_start,
            }

    if best_peak is None or best_tone is None or best_reduction <= 0.0 or best_metadata is None:
        return None

    iq[start:end] -= best_tone
    magnitude = np.abs(iq[start:end])
    iq_after = iq[start:end].copy()

    result = {
        "selected_peak": best_peak,
        "mean_before": original_mean,
        "mean_after": float(np.mean(np.abs(iq_after))),
        "mean_reduction": best_reduction,
        "subtraction_tone": best_tone.copy(),
        "iq_before": segment.copy(),
        "iq_after": iq_after,
    }
    if best_metadata is not None:
        result.update(best_metadata)
    if edge_sample_index is not None:
        result["edge_sample_index"] = edge_sample_index
        result["sample_offset"] = sample_offset
    return result


def apply_lowpass_and_decimate(iq_data, sample_rate, decimation_factor=60,
                               passband_edge=45e3, stopband_edge=55e3,
                               attenuation=80):
    """Apply low-pass FIR filter and decimate the complex IQ data."""
    if stopband_edge <= passband_edge:
        raise ValueError("Stopband edge must be greater than passband edge.")
    if decimation_factor <= 0:
        raise ValueError("Decimation factor must be positive.")

    nyquist = sample_rate / 2.0
    transition_width = (stopband_edge - passband_edge) / nyquist

    if transition_width <= 0:
        raise ValueError("Transition width must be positive.")

    numtaps, beta = signal.kaiserord(attenuation, transition_width)
    beta = float(beta)
    if numtaps % 2 == 0:
        numtaps += 1  # ensure odd number of taps for symmetric FIR

    normalized_passband = passband_edge / nyquist
    fir_coeffs = signal.firwin(numtaps, normalized_passband,
                               window=('kaiser', beta))  # type: ignore[arg-type]

    print(f"\nDesigned low-pass FIR filter:")
    print(f"  Taps: {numtaps}")
    print(f"  Passband edge: {passband_edge / 1e3:.1f} kHz")
    print(f"  Stopband edge: {stopband_edge / 1e3:.1f} kHz")
    print(f"  Estimated attenuation: {attenuation} dB")

    # Apply FIR filter
    filtered = signal.lfilter(fir_coeffs, 1.0, iq_data)

    # Compensate for group delay introduced by FIR filter
    delay = (numtaps - 1) // 2
    if delay > 0:
        filtered = filtered[delay:]

    if len(filtered) < decimation_factor:
        raise ValueError("Not enough samples remain after filtering to decimate.")

    # Trim to an integer number of decimation blocks
    trimmed_length = (len(filtered) // decimation_factor) * decimation_factor
    filtered = filtered[:trimmed_length]

    decimated = filtered[::decimation_factor]
    decimated_sample_rate = sample_rate / decimation_factor

    return decimated, decimated_sample_rate, fir_coeffs


def basic_stats(iq_data, sample_rate):
    """Print basic statistics about the signal"""
    print("\n=== Signal Statistics ===")
    print(f"Total samples: {len(iq_data):,}")
    if sample_rate and sample_rate > 0:
        duration_sec = len(iq_data) / sample_rate
        print(f"Duration: {duration_sec:.6f} seconds")
    else:
        print("Duration: unknown (sample rate not provided)")
    print(f"Mean magnitude: {np.mean(np.abs(iq_data)):.6f}")
    print(f"Max magnitude: {np.max(np.abs(iq_data)):.6f}")
    print(f"RMS magnitude: {np.sqrt(np.mean(np.abs(iq_data)**2)):.6f}")
    if np.mean(np.abs(iq_data)) > 0:
        dynamic_range = 10.0 * np.log10(np.max(np.abs(iq_data)) / np.mean(np.abs(iq_data)))
        print(f"Dynamic range: {dynamic_range:.1f} dB")
    else:
        print("Dynamic range: undefined (mean magnitude is zero)")


if __name__ == "__main__":
    # Load IQ data from file
    iq_data = np.fromfile(path, dtype='complex64')
    # iq_data = iq_data[s_start:s_stop]
    
    # Basic statistics
    basic_stats(iq_data, sample_rate)

    # Apply filtering and decimation to reduce data rate
    print("\nApplying low-pass filter and decimating...")
    iq_decimated, decimated_sample_rate, fir_coeffs = apply_lowpass_and_decimate(
        iq_data,
        sample_rate,
        decimation_factor=decimation_factor,
        passband_edge=passband_edge_hz,
        stopband_edge=stopband_edge_hz,
        attenuation=filter_attenuation_db
    )
    # iq_decimated, decimated_sample_rate = iq_data, sample_rate

    print(f"Decimated sample rate: {decimated_sample_rate:,.2f} Hz")
    basic_stats(iq_decimated, decimated_sample_rate)

    # Use decimated data for analysis
    iq = np.array(iq_decimated, copy=True)
    iq_original = iq.copy()
    analysis_sample_rate = decimated_sample_rate
    reference_pulses = load_pulse_catalog(true_pulse_catalog_path)
    reference_pulses = [
        pulse for pulse in reference_pulses
        if pulse[1] >= t_start and (t_stop <= t_start or pulse[1] <= t_stop)
    ]
    if reference_pulses:
        print(f"Reference pulses available: {len(reference_pulses)}")
    else:
        print("Reference pulse catalog empty or out of range for this segment.")

    # Edge detection using matched filters
    print("\nPerforming edge detection with matched filters...")
    print(f"Template duration: {edge_template_duration_ms:.3f} ms")
    print(f"Peak window: {edge_template_duration_ms:.3f} ms")
    print(f"Edge plots enabled: {enable_edge_plots}")
    
    # Create matched filter template
    rising_template = create_edge_template(analysis_sample_rate)

    # Plot and save the template for visualization
    plot_edge_template(rising_template, analysis_sample_rate)

    edge_fft_results = []
    
    tmp = time.time()
    # Detect edges
    rising_edges, rising_corr, mag_deltas = detect_edges_matched_filter(
        iq,
        analysis_sample_rate,
        rising_template,
    )
    
    print(f"Edge detection time: {time.time() - tmp:.3f}s")

    print("Pre FFT results:", pre_fft_results)

    print(f"\nEdge detection results:")
    print(f"Rising edges detected: {len(rising_edges)}")

    # Timing error analysis vs ground truth
    plot_timing_error_analysis(rising_edges, reference_pulses)

    # Time domain analysis on decimated data with edge overlays
    print("\nPlotting time domain...")
    plot_time_domain(
        iq,
        analysis_sample_rate,
        rising_edges,
        rising_corr,
        original_iq_data=iq_original,
        true_rising_edges=reference_pulses,
        edge_fft_results=edge_fft_results,
        mag_deltas=mag_deltas,
    )
    detected_pulses = []

    if edge_fft_results:
        print("\nSpectral peaks (within 10 dB of max) after rising edges:")
        for result in edge_fft_results:
            print(
                f"  Pulse {result['edge_index']} at {result['edge_time_ms']:.3f} ms "
                f"({len(result['peaks'])} peaks <= {result['threshold_db']:.1f} dB):"
            )
            magnitude_ratio = result.get("magnitude_ratio")
            magnitude_ratio_db = result.get("magnitude_ratio_db")
            if magnitude_ratio is not None and magnitude_ratio_db is not None:
                print(
                    f"    Plateau/baseline ratio: {magnitude_ratio:.3f}x "
                    f"({magnitude_ratio_db:.2f} dB)"
                )
            for peak in result["peaks"]:
                print(
                    f"    {peak['frequency_hz'] / 1e3:.3f} kHz -> "
                    f"amplitude {peak['amplitude']:.3e}, "
                    f"phase {peak['phase_rad']:.3f} rad"
                )
            subtraction = result.get("subtraction")
            if subtraction is not None:
                selected_peak = subtraction.get("selected_peak", {})
                freq_hz = selected_peak.get("frequency_hz")
                freq_text = f"{freq_hz / 1e3:.3f} kHz" if freq_hz is not None else "unknown freq"
                print(
                    f"    Pulse subtracted from signal with {freq_text}: "
                    f"mean {subtraction['mean_before']:.3e} → {subtraction['mean_after']:.3e} "
                    f"(Δ {subtraction['mean_reduction']:.3e})"
                )
            else:
                print("    Pulse not subtracted (no improvement detected)")
            time_domain_file = result.get("time_domain_file")
            if time_domain_file:
                print(f"    Time-domain window saved to '{time_domain_file}'")

            freq_hz = result.get("dominant_freq_hz", 0.0)
            amplitude = result.get("dominant_amplitude", 0.0)
            phase_rad = result.get("dominant_phase_rad", 0.0)
            if subtraction is not None:
                optimized_freq = subtraction.get("optimized_freq_hz")
                if optimized_freq is not None:
                    freq_hz = optimized_freq
                else:
                    freq_hz = subtraction.get("selected_peak", {}).get("frequency_hz", freq_hz)
                amplitude = subtraction.get("selected_peak", {}).get("amplitude", amplitude)
                phase_rad = subtraction.get("optimized_phase_rad",
                                            subtraction.get("phase_at_edge_rad",
                                                            subtraction.get("phase_start_rad", phase_rad)))

            start_time_seconds = result.get("edge_time_ms", 0.0) / 1000.0
            detected_pulses.append((
                len(rising_edges),
                start_time_seconds,
                freq_hz,
                amplitude,
                phase_rad,
            ))

    summarize_pulse_collection("Detections", detected_pulses)

    detected_csv_path = "detected_pulses.csv"
    with open(detected_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["pulse_number", "start_time_seconds", "frequency_hz", "amplitude", "phase_radians"])
        writer.writerows(detected_pulses)



