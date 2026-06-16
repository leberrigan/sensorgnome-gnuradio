"""Boundary-sweep test: does the GRH detector drop pulses depending on where
they land relative to the work()/chunk boundary?

On-device params: samp_rate=48000, pulse_len=2.5ms (120 samp), chunk=2048.
A single clean, high-SNR pulse is swept across sample positions spanning several
chunk boundaries. At each position the signal is fed through detect() chunk by
chunk (exactly like the flowgraph, with the prev_chunk prepend). A correctly
behaving detector should report exactly ONE detection at every position.
"""
import numpy as np, io, contextlib, sys, collections
sys.path.insert(0, ".")
import detect_pulse_2 as dp2

SR        = 48_000
CHUNK     = 2048
PULSE_LEN = int(round(2.5e-3 * SR))   # 120
REDGE     = 4                         # rising-edge ramp ~ decimation-filter rise (matches template)
FOFF      = 2_000.0                    # tone offset, Hz
A         = 1.0                        # pulse amplitude
NOISE     = 0.003                      # ~50 dB SNR -> drops are structural, not marginal
TOTAL     = 5 * CHUNK                  # 10240 samples (~4 interior boundaries)
TOL_S     = 1.0e-3                     # match window: 1 ms

def make_signal(pulse_start, seed=0):
    rng = np.random.default_rng(seed)
    iq = (rng.standard_normal(TOTAL) + 1j*rng.standard_normal(TOTAL)).astype(np.complex64) * NOISE
    idx = np.arange(pulse_start, pulse_start + PULSE_LEN)
    tone = A * np.exp(1j * 2*np.pi * FOFF * idx / SR)
    ramp = np.ones(PULSE_LEN)
    ramp[:REDGE] = np.linspace(0, 1, REDGE)
    iq[pulse_start:pulse_start+PULSE_LEN] += (tone * ramp).astype(np.complex64)
    return iq

def run(pulse_start):
    det = dp2.Pulse_Detector(output_type="test", samp_rate=SR, verbose=False,
                             pulse_len_ms=2.5, high_perf=True)
    iq = make_signal(pulse_start)
    det_ts = []
    for c in range(TOTAL // CHUNK):
        chunk = iq[c*CHUNK:(c+1)*CHUNK]
        with contextlib.redirect_stdout(io.StringIO()):
            out = det.detect([chunk])
        for p in out[0]:
            det_ts.append(float(p.split(',')[0]))
    true_t = pulse_start / SR        # edge ~ pulse_start (detector adds ~third offset, within TOL)
    hits = [t for t in det_ts if abs(t - true_t) < TOL_S + PULSE_LEN/SR]
    return len(det_ts), len(hits)

# Sweep start across boundaries at 2048 and 4096; keep clear of the final held-back tail.
starts = list(range(1500, 4600, 2))
drops, doubles, ok = [], [], 0
for s in starts:
    ntot, nhit = run(s)
    if nhit == 0:
        drops.append(s)
    elif nhit > 1 or ntot > 1:
        doubles.append((s, ntot, nhit))
    else:
        ok += 1

print(f"swept {len(starts)} positions  (SR={SR}, chunk={CHUNK}, pulse_len={PULSE_LEN})")
print(f"  clean single detection : {ok}")
print(f"  DROPPED (0 detections) : {len(drops)}")
print(f"  doubled / extra        : {len(doubles)}")

def near_boundary(s):
    # distance of [pulse_start, pulse_start+PULSE_LEN] region to nearest 2048 multiple
    m = round(s / CHUNK) * CHUNK
    return s - m   # signed offset from nearest boundary

if drops:
    print("\nDROP positions (start, offset-from-nearest-2048-boundary):")
    # compress into ranges
    runs=[]; a=drops[0]; prev=drops[0]
    for s in drops[1:]:
        if s-prev<=2: prev=s
        else: runs.append((a,prev)); a=s; prev=s
    runs.append((a,prev))
    for a,b in runs:
        print(f"  starts {a}..{b}  -> offset {near_boundary(a)}..{near_boundary(b)} (pulse end rel boundary: {a+PULSE_LEN-round(a/CHUNK)*CHUNK})")
if doubles:
    print("\nDOUBLE/EXTRA positions (sample):", [d[0] for d in doubles][:40])
