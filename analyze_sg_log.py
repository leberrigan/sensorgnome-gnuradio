#!/usr/bin/env python3
"""Validate GRH detector changes from a SensorGnome -all.txt log.

Usage:
    python analyze_sg_log.py <file-all.txt> [vah_port grh_port]
    (defaults: vah_port=2 FUNcube/VAH, grh_port=7 AirspyHF/GRH)

Checks, in order:
  1. Row inventory (pulses/bursts/nano per port)
  2. dB-convention self-consistency for GRH ports  (sig - noise == snr)
  3. Sensitivity comparison  (per-pulse SNR + burst min-SNR, VAH vs GRH)
  4. Uptime / co-detection   (bursts each device caught that the other missed)
  5. Absolute-timing offset  (GRH burst ts - VAH burst ts)
  6. For each VAH-only burst, whether GRH detected the underlying pulses
"""
import sys, collections, statistics as st

F = sys.argv[1]
VAH = sys.argv[2] if len(sys.argv) > 2 else '2'
GRH = sys.argv[3] if len(sys.argv) > 3 else '7'
TOL = 1.5            # s, burst co-detection window
BURST_SPAN = 0.35    # s, window to look for a burst's pulses

P = collections.defaultdict(list)   # port -> [(ts,dfreq,sig,noise,snr)]
B = collections.defaultdict(list)   # port -> [(ts,tag,minsnr)]
for line in open(F):
    f = line.strip().split(',')
    if not f or not f[0]:
        continue
    k = f[0][0]; port = f[0][1:]
    try:
        if k == 'p' and len(f) >= 6:
            P[port].append((float(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5])))
        elif k == 'b' and len(f) >= 12:
            # burst fields after 'b<port>': f[1]=ts f[2]=tag ... f[11]=min SNR (sig-noise)
            B[port].append((float(f[1]), int(float(f[2])), float(f[11])))
    except ValueError:
        pass

print(f"=== file: {F}")
print(f"=== comparing VAH port {VAH} vs GRH port {GRH}\n")

# 1. inventory
print("1. ROW INVENTORY")
for port in sorted(set(list(P) + list(B)), key=lambda x: int(x)):
    ts = [p[0] for p in P[port]]
    span = (max(ts) - min(ts)) if len(ts) > 1 else 0
    print(f"   port {port:>2}: {len(P[port]):4d} pulses, {len(B[port]):3d} bursts, span {span:6.0f}s")

# 2. dB self-consistency (GRH): sig - noise should equal snr after the 20log10 fix
print("\n2. dB SELF-CONSISTENCY (GRH sig - noise should == snr)")
for port in (GRH, VAH):
    if not P[port]:
        continue
    diffs = [abs((p[2] - p[3]) - p[4]) for p in P[port]]
    bad = sum(1 for d in diffs if d > 0.5)
    nf = st.median([p[3] for p in P[port]])
    print(f"   port {port:>2}: rows where |sig-noise - snr|>0.5 : {bad}/{len(P[port])}  | median noise {nf:.1f} dB")
print("   PASS: GRH 'bad' ~0  AND GRH median noise on the same (power-dB) scale as VAH")

# 3. sensitivity
print("\n3. SENSITIVITY (higher = more sensitive)")
for port in (VAH, GRH):
    if P[port]:
        psnr = st.median([p[4] for p in P[port]])
        print(f"   port {port:>2}: per-pulse SNR median {psnr:6.2f} dB", end='')
    if B[port]:
        bsnr = st.median([b[2] for b in B[port]])
        print(f"   | burst min-SNR median {bsnr:6.2f} dB")
    else:
        print()
print("   EXPECT: GRH (AirspyHF) burst min-SNR > VAH (FUNcube) by ~8 dB")

# 4. uptime / co-detection
def matched(t, tag, other):
    return any(abs(t - ot) < TOL and tag == otag for ot, otag, _ in other)
b_v, b_g = sorted(B[VAH]), sorted(B[GRH])
only_v = [(t, tag) for t, tag, _ in b_v if not matched(t, tag, b_g)]
only_g = [(t, tag) for t, tag, _ in b_g if not matched(t, tag, b_v)]
both = len(b_v) - len(only_v)
print("\n4. UPTIME / CO-DETECTION")
print(f"   VAH bursts={len(b_v)}  GRH bursts={len(b_g)}  co-detected={both}")
print(f"   VAH-only (GRH MISSED): {len(only_v)}   GRH-only (VAH missed): {len(only_g)}")
print("   PASS: VAH-only ~0 (ignoring at most 1 end-of-recording burst)")

# 5. timing offset
print("\n5. ABSOLUTE-TIMING OFFSET (GRH ts - VAH ts)")
offs = []
for t, tag, _ in b_v:
    c = sorted((abs(ot - t), ot - t) for ot, otag, _ in b_g if otag == tag and abs(ot - t) < TOL)
    if c:
        offs.append(c[0][1])
if offs:
    print(f"   n={len(offs)}  median={st.median(offs)*1e3:8.2f} ms  sd={st.pstdev(offs)*1e3:6.2f} ms"
          f"  min={min(offs)*1e3:.1f} max={max(offs)*1e3:.1f}")
    print("   Track this across buffer changes; smaller = lower pipeline latency.")

# 6. VAH-only bursts: did GRH see the pulses? (drop vs assembly failure)
if only_v:
    print("\n6. VAH-ONLY BURSTS — GRH pulses present nearby?")
    pg = sorted(p[0] for p in P[GRH])
    for t, tag in only_v:
        n = sum(1 for x in pg if t - 0.05 <= x <= t + BURST_SPAN)
        print(f"   t={t:.3f} tag={tag}: GRH pulses in burst window = {n} (4 needed)")
    print("   0 pulses -> dropped upstream; 1-3 -> pulse-level drop; 4 -> burstfinder/timing")
