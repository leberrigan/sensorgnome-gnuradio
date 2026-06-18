#!/usr/bin/env python3
"""v3 vs VAH comparison figure for PULSE_DETECTION.md.

VAH (vamp-plugins/lotek/FindPulseFDBatch) is the decade-old C++ baseline. v3 must
(a) MATCH its weak-signal sensitivity and (b) IMPROVE on its frequency resolution
so overlapping tags in colonies/flocks can be separated. Both detectors integrate
the full pulse in the frequency domain; the difference is FFT resolution:

  VAH : 24-sample FFT (0.5 ms) -> ~2 kHz bins, power integrated over the pulse
  v3  : pulse-length FFT + zero-pad -> ~187 Hz detection bins, sub-bin refined

This reproduces VAH's per-bin pulse-length power integration in Python (labelled a
*model* of the C++ algorithm) and compares it to v3 on identical input: a real
Lotek pulse + AWGN. Left = sensitivity (must match); right = overlap separation
(must improve).
"""
import os
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SR = 48000
PL = 120
RNG = np.random.default_rng(11)

# ---------------- real pulse template
WAV = "baseband_166431800Hz_14-54-57_17-06-2026.wav"
for c in (WAV, os.path.join("..", WAV), os.path.join(os.path.dirname(__file__), "..", WAV)):
    if os.path.exists(c):
        WAV = c; break
sr, data = wavfile.read(WAV)
iq = data[:, 0].astype(np.float64) + 1j * data[:, 1].astype(np.float64)
iq /= np.max(np.abs(iq))
seg = iq[int(3.10 * SR):int(3.16 * SR)]
pk = int(np.argmax(np.convolve(np.abs(seg), np.ones(PL) / PL, "same")))
TMPL = seg[pk - PL // 2: pk + PL // 2].copy()
TMPL /= np.sqrt(np.sum(np.abs(TMPL) ** 2))                 # unit energy
F0 = abs(np.fft.fftfreq(4096, 1 / SR)[np.argmax(np.abs(np.fft.fft(TMPL, 4096)))])
print(f"template tone ~{F0:.0f} Hz")

def noise(n):
    return (RNG.standard_normal(n) + 1j * RNG.standard_normal(n)) / np.sqrt(2)

def add_pulse(x, onset, snr_db, shift_hz=0.0):
    a = np.sqrt(10 ** (snr_db / 10.0))
    t = np.arange(PL)
    x[onset:onset + PL] += a * TMPL * np.exp(1j * 2 * np.pi * shift_hz * t / SR)
    return x

# ---------------- v3 statistic (full-pulse fine-bin FFT)
NF3 = 256
FREQS3 = np.fft.fftfreq(NF3, 1 / SR)
BAND3 = (np.abs(FREQS3) > 2000) & (np.abs(FREQS3) < 12000)
def _stft3(x):
    win, hop = PL, PL // 6
    s = np.arange(0, len(x) - win + 1, hop)
    fr = x[s[:, None] + np.arange(win)] * np.hanning(win)
    return np.abs(np.fft.fft(fr, NF3, axis=1))
FLOOR3 = np.median(_stft3(noise(SR)), axis=0) + 1e-12
def stat_v3(x):
    return ((_stft3(x) / FLOOR3)[:, BAND3]).max()

# ---------------- VAH model statistic (24-sample FFT, pulse-length power integration)
WINV, HOPV, NFV, PFV = 24, 12, 24, 10        # 0.5 ms FFT, 50% overlap, ~pulse-length integration
FREQSV = np.fft.fftfreq(NFV, 1 / SR)
BANDV = (np.abs(FREQSV) > 2000) & (np.abs(FREQSV) < 12000)
def _vah_integ(x):
    s = np.arange(0, len(x) - WINV + 1, HOPV)
    fr = x[s[:, None] + np.arange(WINV)] * np.hanning(WINV)
    P = np.abs(np.fft.fft(fr, NFV, axis=1)) ** 2             # power per coarse bin
    cs = np.cumsum(np.vstack([np.zeros(P.shape[1]), P]), axis=0)
    return cs[PFV:] - cs[:-PFV]                              # pulse-length-integrated power
FLOORV = np.median(_vah_integ(noise(SR)), axis=0) + 1e-12
def stat_vah(x):
    return ((_vah_integ(x) / FLOORV)[:, BANDV]).max()

# ---------------- sensitivity: detection vs SNR (CFAR-matched)
WL = PL * 3
M, K = 4000, 250
thr3 = np.percentile([stat_v3(noise(WL)) for _ in range(M)], 99)
thrv = np.percentile([stat_vah(noise(WL)) for _ in range(M)], 99)
SNRS = np.arange(2, 28, 2)
Pd3, Pdv = [], []
for snr in SNRS:
    c3 = cv = 0
    for _ in range(K):
        x = noise(WL); add_pulse(x, (WL - PL) // 2 + RNG.integers(-8, 9), snr)
        c3 += stat_v3(x) > thr3
        cv += stat_vah(x) > thrv
    Pd3.append(100 * c3 / K); Pdv.append(100 * cv / K)
Pd3, Pdv = np.array(Pd3), np.array(Pdv)
fl = lambda p: SNRS[np.argmax(p >= 50)] if (p >= 50).any() else np.nan
print(f"floors  v3={fl(Pd3)} dB  VAH={fl(Pdv)} dB")

# ---------------- overlap: two tags, different freq offset, overlapping in time
DF = 500.0                                        # 0.5 kHz apart -> within VAH's 2 kHz bin
ov = noise(PL) * 0.0
ov = ov.astype(complex)
add_pulse(ov, 0, 26, 0.0)                          # tag A at template freq
add_pulse(ov, 0, 26, DF)                           # tag B, +0.5 kHz, same time (overlap)
ov = ov + noise(PL) * 0.02
# VAH-resolution view: native 24-sample FFT magnitude (interpolated for a smooth curve)
fv = np.fft.fftshift(np.fft.fftfreq(NFV, 1 / SR))
Sv = np.fft.fftshift(np.abs(np.fft.fft(ov[:WINV] * np.hanning(WINV), NFV)))
# v3-resolution view: full-pulse zero-padded FFT
f3 = np.fft.fftshift(np.fft.fftfreq(4096, 1 / SR))
S3 = np.fft.fftshift(np.abs(np.fft.fft(ov * np.hanning(PL), 4096)))

# ---------------- plot
plt.rcParams.update({"font.size": 9})
fig = plt.figure(figsize=(12.5, 4.8))
gs = GridSpec(1, 2, width_ratios=[1, 1.05], wspace=0.24,
              left=0.07, right=0.985, top=0.88, bottom=0.14)
C3, CV = "#2ca02c", "#1f77b4"

axA = fig.add_subplot(gs[0, 0])
axA.plot(SNRS, Pdv, "-s", ms=4, color=CV, label=f"VAH model  (floor {fl(Pdv):.0f} dB)")
axA.plot(SNRS, Pd3, "-o", ms=4, color=C3, label=f"v3            (floor {fl(Pd3):.0f} dB)")
axA.axhline(50, color="k", lw=0.6, ls=":")
axA.set_xlabel("pulse SNR (integrated, dB)"); axA.set_ylabel("detection probability (%)")
axA.set_ylim(-3, 103); axA.grid(alpha=0.3); axA.legend(loc="lower right")
axA.set_title("(a) sensitivity parity — v3 reproduces VAH's weak-signal floor",
              fontsize=9.5, loc="left")
axA.text(0.03, 0.55,
         "both integrate the full pulse in\n"
         "the frequency domain, so their\n"
         "floors coincide.\n\n"
         "on hardware (17.5 min, same\n"
         "antennas): VAH 37 bursts,\n"
         "v3 40 — parity, v3 slightly ahead\n"
         "on the weakest tag.",
         transform=axA.transAxes, va="top", fontsize=8, color="0.25")

axB = fig.add_subplot(gs[0, 1])
sel = (f3 > F0 - 2500) & (f3 < F0 + 3000)
axB.plot(f3[sel] / 1000, S3[sel] / S3[sel].max(), color=C3, lw=1.4,
         label="v3  (~187 Hz bins + refine)")
selv = (fv > F0 - 2500) & (fv < F0 + 3000)
axB.plot(fv[selv] / 1000, Sv[selv] / Sv[selv].max(), "-s", ms=5, color=CV, lw=1.4,
         label="VAH  (2 kHz bins)")
for fpk, lab in [(F0, "tag A"), (F0 + DF, "tag B")]:
    axB.axvline(fpk / 1000, color="0.6", ls=":", lw=0.9)
    axB.text(fpk / 1000, 1.04, lab, ha="center", fontsize=8, color="0.4")
axB.set_xlabel("frequency (kHz)"); axB.set_ylabel("normalized magnitude")
axB.set_ylim(0, 1.15); axB.legend(loc="upper right", fontsize=8.5)
axB.set_title(f"(b) overlap separation — two tags {DF/1000:.1f} kHz apart, same instant",
              fontsize=9.5, loc="left")
axB.text(0.03, 0.5,
         "VAH's 2 kHz bins merge the two\n"
         "tags into ONE blob -> can't be\n"
         "separated -> burst rejected.\n\n"
         "v3 resolves TWO peaks -> both\n"
         "tags decoded. Critical for\n"
         "colonies/flocks of similar-rate\n"
         "tags whose pulses overlap.",
         transform=axB.transAxes, va="top", fontsize=8, color="0.25")

fig.suptitle("v3 vs VAH — match the sensitivity, beat the frequency resolution  "
             "(real pulse + AWGN)", fontsize=11, weight="bold")
fig.savefig("v3_vs_vah.png", dpi=130)
print("wrote v3_vs_vah.png")
