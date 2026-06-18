#!/usr/bin/env python3
"""Generate the pulse-detection comparison figure for PULSE_DETECTION.md.

Uses a REAL Lotek pulse (extracted from a FunCube IQ recording) re-inserted into
additive white Gaussian noise at controlled SNR. Each detector generation is
represented by its core detection statistic:

  v1  amplitude   - peak of the smoothed |IQ| envelope (no frequency selectivity)
  v2  edge        - peak normalized cross-correlation with a rising-edge template
  v3 / VAH  tone  - peak per-bin power integrated over the pulse (STFT), i.e. the
                    frequency-domain full-pulse integration VAH also uses

For a FAIR comparison every detector's threshold is set by CFAR: tuned on
noise-only windows to the same false-alarm probability (1% per window). The left
panels show how each statistic responds to one real burst at a single SNR; the
right panel sweeps SNR and plots detection probability -> the sensitivity floor.
"""
import os
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SR = 48000
PL = 120                     # pulse length in samples (2.5 ms @ 48 kHz)
RNG = np.random.default_rng(7)

# ---------------------------------------------------------------- real pulse
WAV = "baseband_166431800Hz_14-54-57_17-06-2026.wav"
for cand in (WAV, os.path.join("..", WAV), os.path.join(os.path.dirname(__file__), "..", WAV)):
    if os.path.exists(cand):
        WAV = cand; break
sr, data = wavfile.read(WAV)
iq = (data[:, 0].astype(np.float64) + 1j * data[:, 1].astype(np.float64))
iq /= np.max(np.abs(iq))
seg = iq[int(3.10 * SR):int(3.16 * SR)]                 # around the tag-732 pulse
sm = np.convolve(np.abs(seg), np.ones(PL) / PL, "same")
pk = int(np.argmax(sm))
TMPL = seg[pk - PL // 2: pk + PL // 2].copy()
TMPL /= np.sqrt(np.sum(np.abs(TMPL) ** 2))              # unit energy
print(f"extracted real pulse template: {len(TMPL)} samples, tone ~"
      f"{abs(np.fft.fftfreq(256,1/SR)[np.argmax(np.abs(np.fft.fft(TMPL,256)))]):.0f} Hz")

# rising-edge template for v2 (same construction as detect_pulse_2)
def edge_template(edge_ms=0.3):
    s = max(1, int(round(edge_ms * 1e-3 * SR / 3))); tot = max(3, s * 3)
    t = np.zeros(tot); t[s:2 * s] = np.linspace(0, 1, s); t[2 * s:] = 1.0
    t -= t.mean(); t /= np.linalg.norm(t); return t
EDGE = edge_template()

# ---------------------------------------------------------------- signal model
def noise(n):
    return (RNG.standard_normal(n) + 1j * RNG.standard_normal(n)) / np.sqrt(2)  # var 1/complex samp

def add_pulse(x, onset, snr_db):
    a = np.sqrt(10 ** (snr_db / 10.0))                 # unit-energy tmpl -> integrated SNR
    x[onset:onset + PL] += a * TMPL
    return x

# ---------------------------------------------------------------- statistics
def trace_v1(x):
    return np.convolve(np.abs(x), np.ones(PL) / PL, "valid")           # smoothed amplitude

def trace_v2(x):                                                        # edge matched filter
    return np.correlate(np.abs(x), EDGE, "valid")                       # projection onto rising step

NFFT = 256
FREQS = np.fft.fftfreq(NFFT, 1 / SR)
BAND = (np.abs(FREQS) > 2000) & (np.abs(FREQS) < 12000)                 # tag band, like the real detector

def _stft_mag(x):
    win, hop = PL, PL // 6
    starts = np.arange(0, len(x) - win + 1, hop)
    fr = x[starts[:, None] + np.arange(win)] * np.hanning(win)
    return starts, np.abs(np.fft.fft(fr, NFFT, axis=1))

# stable per-bin noise floor from a long noise reference (the real detector takes
# the per-bin median over a ~1 s chunk, where any pulse is a negligible fraction)
_, _mfloor = _stft_mag(noise(SR))
GLOBAL_FLOOR = np.median(_mfloor, axis=0) + 1e-12

def stft_snr(x):
    starts, mag = _stft_mag(x)
    return starts, (mag / GLOBAL_FLOOR)[:, BAND]                        # in-band per-bin SNR

def trace_v3(x):
    _, snr = stft_snr(x); return snr.max(axis=1)                        # per-frame best-bin SNR

def stat_v1(x): return trace_v1(x).max()
def stat_v2(x): return trace_v2(x).max()
def stat_v3(x): _, s = stft_snr(x); return s.max()

# ---------------------------------------------------------------- CFAR thresholds
WL = PL * 3                                  # compact window: comparable search space for all
M = 4000
ns = {"v1": [], "v2": [], "v3": []}
for _ in range(M):
    x = noise(WL)
    ns["v1"].append(stat_v1(x)); ns["v2"].append(stat_v2(x)); ns["v3"].append(stat_v3(x))
THR = {k: np.percentile(v, 99) for k, v in ns.items()}                  # 1% false alarm / window
print("CFAR thresholds (1%/window):", {k: round(v, 3) for k, v in THR.items()})

# ---------------------------------------------------------------- detection vs SNR
SNRS = np.arange(-2, 34, 2)
K = 200
Pd = {"v1": [], "v2": [], "v3": []}
for snr in SNRS:
    c = {"v1": 0, "v2": 0, "v3": 0}
    for _ in range(K):
        x = noise(WL); add_pulse(x, (WL - PL) // 2 + RNG.integers(-8, 9), snr)
        if stat_v1(x) > THR["v1"]: c["v1"] += 1
        if stat_v2(x) > THR["v2"]: c["v2"] += 1
        if stat_v3(x) > THR["v3"]: c["v3"] += 1
    for k in Pd: Pd[k].append(c[k] / K)
Pd = {k: np.array(v) for k, v in Pd.items()}
def floor50(p):
    i = np.argmax(p >= 0.5)
    return SNRS[i] if p[i] >= 0.5 else np.nan
print("50% detection floor (dB):", {k: floor50(v) for k, v in Pd.items()})

# ---------------------------------------------------------------- mechanism burst
DEMO_SNR = 16
def pd_at(k):
    return float(np.interp(DEMO_SNR, SNRS, Pd[k]) * 100)
GAPS = [0.022, 0.0975, 0.0538]            # tag-17
onsets_t = np.cumsum([0.04] + GAPS)
xb = noise(int(0.32 * SR))
for ot in onsets_t:
    add_pulse(xb, int(ot * SR), DEMO_SNR)
tb = np.arange(len(xb)) / SR * 1000        # ms

# ---------------------------------------------------------------- plot
plt.rcParams.update({"font.size": 9})
fig = plt.figure(figsize=(13, 7.2))
gs = GridSpec(4, 2, width_ratios=[1.25, 1], hspace=0.45, wspace=0.22,
              left=0.06, right=0.985, top=0.92, bottom=0.09)
COL = {"v1": "#d62728", "v2": "#ff7f0e", "v3": "#2ca02c"}

# (a) spectrogram
axA = fig.add_subplot(gs[0, 0])
axA.specgram(xb, NFFT=128, Fs=SR, noverlap=112, cmap="magma")
axA.set_ylim(0, 12000); axA.set_ylabel("freq (Hz)")
axA.set_title(f"(a) one real tag-17 burst in noise @ {DEMO_SNR} dB  —  4 tone pulses",
              fontsize=9, loc="left")
axA.set_xticklabels([])
for ot in onsets_t:
    axA.axvline(ot, color="cyan", lw=0.8, ls=":", alpha=0.7)

def panel(ax, t, y, thr, color, label, title, pd):
    ax.plot(t, y, color=color, lw=0.9)
    ax.axhline(thr, color="k", lw=0.8, ls="--")
    ax.text(0.995, 0.9, "detect threshold", ha="right", va="top",
            transform=ax.transAxes, fontsize=7, color="k")
    for ot in onsets_t:
        ax.axvspan(ot * 1000, (ot + PL / SR) * 1000, color=color, alpha=0.10)
    ax.set_title(f"{title}   —   detects {pd:.0f}% of pulses @ {DEMO_SNR} dB",
                 fontsize=8.5, loc="left")
    ax.set_ylabel(label)

axB = fig.add_subplot(gs[1, 0])
y1 = trace_v1(xb); t1 = (np.arange(len(y1)) + PL // 2) / SR * 1000
panel(axB, t1, y1, THR["v1"], COL["v1"], "|IQ| avg", "(b) v1 amplitude envelope", pd_at("v1"))
axB.set_xticklabels([])

axC = fig.add_subplot(gs[2, 0])
y2 = trace_v2(xb); t2 = (np.arange(len(y2)) + len(EDGE) // 2) / SR * 1000
panel(axC, t2, y2, THR["v2"], COL["v2"], "edge MF", "(c) v2 rising-edge match", pd_at("v2"))
axC.set_xticklabels([])

axD = fig.add_subplot(gs[3, 0])
starts, _ = stft_snr(xb); y3 = trace_v3(xb); t3 = (starts + PL // 2) / SR * 1000
panel(axD, t3, y3, THR["v3"], COL["v3"], "tone SNR", "(d) v3 / VAH  full-pulse tone SNR", pd_at("v3"))
axD.set_xlabel("time (ms)")

# (e) detection vs SNR
axE = fig.add_subplot(gs[:, 1])
names = {"v1": "v1  amplitude", "v2": "v2  edge", "v3": "v3 / VAH  tone"}
for k in ("v1", "v2", "v3"):
    axE.plot(SNRS, 100 * Pd[k], "-o", ms=3, color=COL[k], label=names[k])
axE.axhline(50, color="k", lw=0.6, ls=":")
axE.axvline(DEMO_SNR, color="0.5", lw=0.8, ls=":")
axE.text(DEMO_SNR + 0.3, 4, f"panels (a–d)\n@ {DEMO_SNR} dB", fontsize=7, color="0.4")
axE.set_xlabel("pulse SNR (integrated, dB)"); axE.set_ylabel("detection probability (%)")
axE.set_title("(e) sensitivity: detection vs SNR\n(same 1%/window false-alarm rate)",
              fontsize=9, loc="left")
axE.set_ylim(-3, 103); axE.grid(alpha=0.3); axE.legend(loc="lower right", fontsize=8.5)
fl = {k: floor50(Pd[k]) for k in Pd}
txt = "50% detection floor:\n" + "\n".join(
    f"  {names[k]}:  {fl[k]:.0f} dB" for k in ("v1", "v2", "v3"))
axE.text(0.03, 0.97, txt, transform=axE.transAxes, va="top", fontsize=8,
         bbox=dict(boxstyle="round", fc="white", ec="0.7"))
axE.text(0.03, 0.62,
         "v2's short edge has less processing\n"
         "gain than v1 — it traded raw\n"
         "sensitivity for shape-robustness,\n"
         "overlap handling & frequency.\n"
         "v3 recovers sensitivity (full tone).",
         transform=axE.transAxes, va="top", fontsize=7.5, color="0.25")

fig.suptitle("How each detector generation 'sees' a Lotek pulse  (real pulse + AWGN)",
             fontsize=11, weight="bold")
out = "pulse_detection_comparison.png"
fig.savefig(out, dpi=130)
print("wrote", out)
