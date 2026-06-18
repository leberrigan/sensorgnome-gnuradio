# detect_pulse_3.py — STFT / spectrogram tone detector for SensorGnome
# Lucas Berrigan 2026
#
# A ground-up alternative to detect_pulse_2's edge-correlation approach.
#
# WHY: a Lotek pulse is a ~2.5 ms CW tone. detect_pulse_2 finds the rising EDGE
# (a 3-5 sample transition) — the lowest-SNR part of the pulse, which caps its
# sensitivity at ~15-20 dB. This detector finds the TONE instead:
#
#   * Short-time Fourier transform turns each pulse into a bright, ~constant-
#     frequency blob in the spectrogram. The tone integrates coherently over the
#     whole pulse window => full-pulse (VAH-level) weak-signal sensitivity.
#   * Overlapping tags at different frequencies land in different bins, so they
#     separate automatically -- no tone subtraction needed (the thing that made
#     detect_pulse_2 complex).
#   * Frequency is read from the peak bin and refined by phase regression over the
#     pulse samples => accurate per-pulse frequency offsets, which matter for tag
#     IDs and especially for telling overlapping pulses apart.
#
# Sensitivity is a knob (`sensitivity=`): "max" is the lowest threshold; raise it
# in noisy environments. Robustness comes mostly from the DURATION constraint —
# a real pulse is a tone that persists ~2.5 ms, whereas a noise spike is brief —
# so we can keep the per-frame threshold low without flooding false positives.
#
# Output matches detect_pulse_2 (drop-in for gr_detect_pulses / burstfinder):
#   p<port>,ts,dfreq_kHz,sig_db,noise_db,snr_db,dur_ms,nsamp,2048,0,0,phase

import time
import numpy as np
from scipy import ndimage


class Pulse_Detector:

    # Per-frame bin-SNR threshold (dB above the per-bin noise floor) by mode.
    # Calibrated on real IQ: ~18 dB is the noise-floor-clean point (0 false
    # positives against thermal noise) while giving best weak-tag recovery; lower
    # than that starts admitting noise blobs (and the dense mask gets slow).
    # NB: these reject NOISE, not tonal interference (e.g. voice on-frequency) —
    # the STFT detects any tone, so co-channel tones need a shape filter or RF fix.
    SENSITIVITY = {"max": 18.0, "high": 22.0, "medium": 26.0, "low": 30.0}

    def __init__(self, output_type="test", verbose=True, filename="output.csv", port=0,
                 samp_rate=48000, min_snr_db=6, debounce_samples=10, pulse_len_ms=2.5,
                 pulse_padding_s=2.5e-3, freq_min=None, freq_max=None, high_perf=True,
                 sensitivity="max"):
        self.output_type = output_type
        self.verbose = bool(verbose)
        self.filename = filename
        self.port = int(port)
        self.samp_rate = int(samp_rate)
        self.min_snr_db = float(min_snr_db)
        self.pulse_len_ms = float(pulse_len_ms)
        self.pulse_len_samples = max(8, int(pulse_len_ms * 1e-3 * self.samp_rate))
        self.freq_min = freq_min
        self.freq_max = freq_max

        # --- STFT parameters ---
        # Window ~ pulse length => near matched-filter integration (max sensitivity).
        # Fine hop => time resolution for pulse timing and boundary handling.
        self.stft_win = self.pulse_len_samples
        self.stft_hop = max(2, self.stft_win // 6)
        self.nfft = 1 << int(np.ceil(np.log2(self.stft_win * 2)))   # zero-pad x2 -> finer bins
        self.window = np.hanning(self.stft_win).astype(np.float64)
        self.win_sum = float(np.sum(self.window))
        self.freqs = np.fft.fftfreq(self.nfft, d=1.0 / self.samp_rate)

        # --- detection gates ---
        self.det_threshold_db = self.SENSITIVITY.get(sensitivity, 8.0)
        self._det_factor = 10.0 ** (self.det_threshold_db / 20.0)
        # a pulse-blob must persist roughly a pulse-length (in frames); the window
        # smears a pulse to ~(pulse+window) in the spectrogram, hence the wide max.
        self.min_pulse_frames = max(1, int(0.4 * self.pulse_len_samples / self.stft_hop))
        self.max_pulse_frames = int((3.0 * self.pulse_len_samples + self.stft_win) / self.stft_hop)
        self.max_blob_bins = max(4, self.nfft // 8)   # reject broadband blobs (not a tone)

        # --- broadband-flood rejection (window level, not pulse level) ---
        # A single pulse can't be told apart from a noise blob, but a FLOOD can:
        # wideband interference (switching/electrical noise) fires many pulses at
        # many DIFFERENT frequencies within a short window. A real tag is 4 pulses
        # at ONE frequency, and even several simultaneous tags share only a few
        # frequencies -- so dropping pulses that sit amid >flood_distinct_freqs
        # distinct nearby frequencies discards floods without touching real tags.
        # Floods are a rare, transient situation we don't care to detect through.
        self.flood_window_s = 0.20            # look this far either side of a pulse
        self.flood_freq_sep = 300.0           # Hz; freqs >= this apart are "distinct" tags
        self.flood_distinct_freqs = 6         # drop if more distinct nearby freqs than this
        # ...and drop on sheer density too: a real burst is <=4 pulses (<=~8 for two
        # tags at once) within flood_window_s, but a voice/data flood -- even though
        # narrowband (few distinct freqs) -- fires dozens. Calibrated on a voice-flood
        # recording: real tags sit at 3-5 nearby pulses, the flood at ~73; 12 cleanly
        # separates them (catches >98% of flood, drops 0 tag pulses).
        self.flood_max_pulses = 12            # drop if more nearby pulses than this

        # --- streaming state ---
        self.carry = self.stft_win + 2 * self.pulse_len_samples
        self.prev_tail = None
        self.sample_counter = 0
        self.reference_time = None
        self.time_chunk_start = None
        self.recent_emitted_abs = []        # absolute onset samples, for cross-chunk dedup
        self.dedup_tol = self.stft_hop
        # Re-anchor the sample-clock to wall time if it drifts past this (dropped
        # samples step every later timestamp behind real time). Mirrors detect_pulse_2.
        self.time_error_threshold = 0.1     # seconds

        # --- flood detection: deferred across chunks ---
        # GnuRadio hands work() small, variable buffers, so the flood test cannot
        # be done per-chunk (its +-flood_window_s neighbourhood spans many chunks).
        # Instead candidate pulses are buffered and a pulse is only decided once its
        # full forward window has arrived (a ~flood_window_s emission latency).
        self._pending = []                  # [string, abs_onset, ts, freq, decided]
        self.flood_open = False
        self.flood_start = 0.0
        self.flood_end = 0.0
        self.flood_dropped = 0

        if self.verbose:
            print("port, ts, dfreq_kHz, sig_db, noise_db, snr_db, dur_ms, nsamp, ...")

    # ------------------------------------------------------------------
    def detect(self, input):
        raw_iq = np.asarray(input[0], dtype=np.complex64)
        n = len(raw_iq)
        output = []

        if self.output_type == "test":
            self.reference_time = 0.0
        elif self.reference_time is None:
            self.reference_time = time.time()

        if self.prev_tail is not None and len(self.prev_tail) > 0:
            buf = np.concatenate([self.prev_tail, raw_iq]).astype(np.complex64)
            carry_n = len(self.prev_tail)
        else:
            buf = raw_iq
            carry_n = 0

        self.time_chunk_start = self.reference_time + self.sample_counter / self.samp_rate

        # Resync the sample-clock to wall time on drift (e.g. a scheduler overflow
        # drops samples, so sample_counter lags and every later timestamp steps
        # behind). Flush the deferred flood buffer at the OLD clock first, else a
        # pending pulse would be stranded with a now-future timestamp.
        if self.output_type != "test":
            now = time.time()
            if abs(now - (self.time_chunk_start + n / self.samp_rate)) > self.time_error_threshold:
                self._process(float("inf"), output, final=True)
                if self.flood_open:
                    self._emit_flood(output)
                self.sample_counter = 0
                self.reference_time = now - n / self.samp_rate
                self.time_chunk_start = self.reference_time
                self.recent_emitted_abs = []

        buf_abs_start = self.sample_counter - carry_n

        for r in self._detect(buf, buf_abs_start, carry_n):
            self._pending.append([r[0], r[1], r[2], r[3], False])

        # decide pulses whose full forward window has arrived (now - flood_window_s)
        chunk_end_ts = self.time_chunk_start + (len(buf) - carry_n) / self.samp_rate
        self._process(chunk_end_ts - self.flood_window_s, output, final=False)

        # prune dedup list to the carried-over window
        buf_abs_end = buf_abs_start + len(buf)
        keep_from = buf_abs_end - self.carry - self.dedup_tol
        self.recent_emitted_abs = [e for e in self.recent_emitted_abs if e >= keep_from]

        self.prev_tail = buf[-self.carry:].copy() if len(buf) >= self.carry else buf.copy()
        self.sample_counter += n

        if self.output_type == "test":
            return output, None, None, None
        return n

    # ------------------------------------------------------------------
    def _process(self, horizon_ts, output, final):
        """Flood-test and emit every pending pulse at or before horizon_ts (its
        +-flood_window_s neighbours are all present by now). Undecided pulses are
        kept in the buffer; decided ones are retained a little longer so they
        still provide neighbour context to later pulses, then pruned."""
        if not self._pending:
            return
        self._pending.sort(key=lambda r: r[2])
        ts_all = np.array([r[2] for r in self._pending])
        fr_all = np.array([r[3] for r in self._pending])
        dropped_ts = []
        for i, rec in enumerate(self._pending):
            if rec[4]:                                   # already decided
                continue
            ti = ts_all[i]
            if not final and ti > horizon_ts:            # forward window not complete yet
                continue
            rec[4] = True
            near = np.abs(ts_all - ti) < self.flood_window_s
            near_count = int(np.count_nonzero(near))
            nf = np.sort(fr_all[near])
            distinct = 1 + int(np.count_nonzero(np.diff(nf) >= self.flood_freq_sep))
            # flood = many distinct nearby freqs (broadband) OR sheer pulse density
            # (narrowband voice/data trains). Either way it isn't a tag burst.
            if distinct > self.flood_distinct_freqs or near_count > self.flood_max_pulses:
                dropped_ts.append(ti)
                continue
            p, abs_onset = rec[0], rec[1]
            if any(abs(abs_onset - e) <= self.dedup_tol for e in self.recent_emitted_abs):
                continue
            self.recent_emitted_abs.append(abs_onset)
            if self.output_type == "stream":
                print(f"p{self.port},{p}", flush=True)
            output.append(p)

        self._track_flood(sorted(dropped_ts), output)

        # keep decided pulses one more window for neighbour context, then drop
        keep_before = horizon_ts - self.flood_window_s
        self._pending = [r for r in self._pending if not (r[4] and r[2] < keep_before)]

    # ------------------------------------------------------------------
    def _track_flood(self, dropped_ts, output):
        """Merge dropped-pulse timestamps into flood intervals (gaps shorter than
        flood_window_s are one flood) and emit a marker line (F<port>,start,end,
        n_dropped) whenever a flood ends."""
        for t in dropped_ts:
            if self.flood_open and (t - self.flood_end) <= self.flood_window_s:
                self.flood_end = t
                self.flood_dropped += 1
            else:
                if self.flood_open:
                    self._emit_flood(output)
                self.flood_open = True
                self.flood_start = t
                self.flood_end = t
                self.flood_dropped = 1

    def _emit_flood(self, output):
        self.flood_open = False
        line = (f"F{self.port},{self.flood_start:.6f},{self.flood_end:.6f},"
                f"{self.flood_dropped}")
        if self.output_type == "stream":
            print(line, flush=True)
        else:
            output.append(line)

    def flush(self):
        """Decide any buffered pulses and emit a still-open flood interval (call on
        clean shutdown / end of stream). Returns emitted lines (test mode)."""
        out = []
        self._process(float("inf"), out, final=True)
        if self.flood_open:
            self._emit_flood(out)
        return out

    # ------------------------------------------------------------------
    def _detect(self, buf, buf_abs_start, carry_n):
        win, hop = self.stft_win, self.stft_hop
        if len(buf) < win + hop:
            return []

        # --- STFT (vectorised) ---
        starts = np.arange(0, len(buf) - win + 1, hop)
        frames = buf[starts[:, None] + np.arange(win)[None, :]] * self.window[None, :]
        mag = np.abs(np.fft.fft(frames, self.nfft, axis=1)).astype(np.float64)   # (nframes, nbins)

        # --- per-bin noise floor (median over time; robust to sparse pulses) ---
        floor = np.median(mag, axis=0) + 1e-12
        mask = mag > (floor * self._det_factor)[None, :]
        if not mask.any():
            return []

        # --- connected components: each pulse = a blob (time x freq) ---
        labels, nlab = ndimage.label(mask, structure=np.ones((3, 3)))

        # don't emit pulses whose body runs past the end of buf (defer to next chunk),
        # and don't re-emit pulses sitting in the carried-over region (minus a margin).
        end_guard = len(buf) - self.pulse_len_samples
        results = []
        for lab in range(1, nlab + 1):
            fr, bn = np.where(labels == lab)
            dur_frames = fr.max() - fr.min() + 1
            if dur_frames < self.min_pulse_frames or dur_frames > self.max_pulse_frames:
                continue
            if (bn.max() - bn.min() + 1) > self.max_blob_bins:
                continue                                   # broadband => not a tag tone
            blob_mag = mag[fr, bn]
            pk = int(np.argmax(blob_mag))
            pk_frame, pk_bin, peak_mag = fr[pk], bn[pk], blob_mag[pk]

            onset_sample = int(starts[pk_frame])           # best-aligned window start ~ pulse onset
            # Lower bound must meet the PREVIOUS chunk's end_guard cutoff exactly, or
            # a (pulse_len - hop)-sample gap opens at every chunk boundary and pulses
            # landing there are detected by neither chunk (severe with small GR
            # buffers). carry_n - pulse_len_samples closes the gap; recent_emitted_abs
            # dedup absorbs the small overlap.
            if onset_sample < carry_n - self.pulse_len_samples or onset_sample > end_guard:
                continue

            coarse_freq = float(self.freqs[pk_bin])
            freq, phase0 = self._refine_freq(buf, onset_sample, coarse_freq)
            if self.freq_min is not None and freq < self.freq_min:
                continue
            if self.freq_max is not None and freq > self.freq_max:
                continue

            noise_mag = float(floor[pk_bin])
            snr_db = 20.0 * np.log10(peak_mag / noise_mag)         # integrated (VAH-style) SNR
            if snr_db < self.min_snr_db:
                continue
            sig_db = 20.0 * np.log10(peak_mag / self.win_sum)
            noise_db = 20.0 * np.log10(noise_mag / self.win_sum)
            dur_ms = dur_frames * hop / self.samp_rate * 1e3
            nsamp = int(dur_frames * hop)
            ts = self.time_chunk_start + (onset_sample - carry_n) / self.samp_rate
            abs_onset = buf_abs_start + onset_sample
            results.append((
                f"{ts:.6f},{freq/1e3:.3f},{sig_db:.2f},{noise_db:.2f},{snr_db:.2f},"
                f"{dur_ms:.3f},{nsamp},2048,0,0,{phase0:.6f}",
                abs_onset,
                ts,
                freq,
            ))
        results.sort(key=lambda r: r[1])
        return results          # raw [string, abs_onset, ts, freq]; flood-tested in _process
        return kept, None

    def _refine_freq(self, buf, onset_sample, coarse_freq):
        """Accurate, *consistent* per-pulse frequency.

        A high-resolution (zero-padded) FFT over the whole pulse, with the peak
        searched only near the detected tone (so a stronger co-channel tone can't
        steal it), then quadratic (log-magnitude) interpolation for sub-bin
        accuracy. This integrates the full pulse, so repeated pulses of the same
        tag give the same answer -> low within-burst freqsd (phase regression over
        a short segment was too noisy, ~0.6 kHz spread, which tripped burstfinder's
        0.1 kHz FREQ_SLOP and got the bursts rejected).
        """
        n = self.pulse_len_samples
        a = max(0, onset_sample)
        b = min(len(buf), onset_sample + n)
        seg = buf[a:b]
        if len(seg) < 16:
            return coarse_freq, 0.0
        seg = seg * np.hanning(len(seg))
        nf = 4096
        spec = np.fft.fft(seg, nf)
        mag = np.abs(spec)
        ff = np.fft.fftfreq(nf, d=1.0 / self.samp_rate)
        det_bin = self.samp_rate / self.nfft           # detection STFT bin width
        band = (ff > coarse_freq - 2 * det_bin) & (ff < coarse_freq + 2 * det_bin)
        if not band.any():
            band = np.ones(nf, dtype=bool)
        k = int(np.argmax(np.where(band, mag, 0.0)))
        # quadratic interpolation on log-magnitude (sub-bin)
        km1, kp1 = (k - 1) % nf, (k + 1) % nf
        a0, b0, c0 = np.log(mag[km1] + 1e-20), np.log(mag[k] + 1e-20), np.log(mag[kp1] + 1e-20)
        denom = a0 - 2 * b0 + c0
        delta = 0.5 * (a0 - c0) / denom if denom != 0 else 0.0
        delta = max(-0.5, min(0.5, delta))
        freq = float(ff[k] + delta * (self.samp_rate / nf))
        return freq, float(np.angle(spec[k]))

    def __del__(self):
        pass
