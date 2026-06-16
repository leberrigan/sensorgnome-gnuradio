"""Simulate the streaming clock to verify startup-drain anchoring.

Models a pipeline with a large STARTUP latency (0.14 s prefill) and a small
steady USB latency L=5 ms. Sample s is captured at wall s/sr and becomes
available to detect() at wall s/sr + L; the first detect() call doesn't happen
until 0.14 s (device/driver startup), so a backlog has accumulated. We drive
detect() with a fake clock following that schedule and check that, after the
drain anchors, an emitted pulse's timestamp equals its TRUE capture time + L
(~5 ms) — NOT the 0.14 s startup latency.
"""
import numpy as np, io, contextlib, sys
sys.path.insert(0, ".")
import detect_pulse_2 as dp2

SR = 48000; CHUNK = 2048; L = 0.005; STARTUP = 0.14; PROC = 0.0005
REDGE = 4; FOFF = 2000.0
FAKE_NOW = [0.0]
dp2.time.time = lambda: FAKE_NOW[0]      # patch the module clock

PULSE_CHUNK = 7; PULSE_LOCAL = 500
G_pulse = PULSE_CHUNK * CHUNK + PULSE_LOCAL          # global sample index of pulse start
true_capture = G_pulse / SR                          # true wall time it was captured

def build_chunk(k):
    rng = np.random.default_rng(k)
    iq = (rng.standard_normal(CHUNK) + 1j*rng.standard_normal(CHUNK)).astype(np.complex64) * 0.003
    if k == PULSE_CHUNK:
        idx = np.arange(PULSE_LOCAL, PULSE_LOCAL + 120)
        tone = np.exp(1j * 2*np.pi * FOFF * (G_pulse + np.arange(120)) / SR)
        ramp = np.ones(120); ramp[:REDGE] = np.linspace(0, 1, REDGE)
        iq[PULSE_LOCAL:PULSE_LOCAL+120] += (tone * ramp).astype(np.complex64)
    return iq

det = dp2.Pulse_Detector(output_type="stream", samp_rate=SR, verbose=True,
                         pulse_len_ms=2.5, high_perf=True, port=7)

prev_wall = 0.0
emitted = []
log = io.StringIO()
for k in range(12):
    avail = ((k+1)*CHUNK - 1)/SR + L          # newest sample of chunk k available
    call_wall = max(prev_wall + PROC, avail, STARTUP if k == 0 else 0)
    FAKE_NOW[0] = call_wall
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        det.detect([build_chunk(k)])
    for line in buf.getvalue().splitlines():
        if line.startswith("p7,"):
            emitted.append(float(line.split(",")[1]))
        if "anchored" in line:
            log.write(f"  [k={k} wall={call_wall*1e3:.0f}ms] {line.strip()}\n")
    prev_wall = call_wall

print(log.getvalue(), end="")
print(f"\npulse true capture time : {true_capture*1000:.2f} ms  (global sample {G_pulse})")
print(f"steady USB latency L    : {L*1000:.2f} ms  (irreducible)")
print(f"old-style startup offset: {STARTUP*1000:.0f} ms  (what we're removing)")
if emitted:
    for ts in emitted:
        off = (ts - true_capture) * 1000
        print(f"emitted ts {ts*1000:.2f} ms  -> offset from true capture = {off:+.2f} ms")
    off = (emitted[0] - true_capture) * 1000
    ok = abs(off - L*1000) < 5      # within a few ms of L (edge 'third' offset etc.)
    print(f"\nRESULT: offset {off:+.2f} ms vs expected ~{L*1000:.0f} ms  -> "
          f"{'PASS (startup latency removed)' if ok else 'FAIL'}")
else:
    print("FAIL: no pulse emitted")
