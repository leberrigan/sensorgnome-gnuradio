## Gnu Radio host, flow graphs, and blocks for Sensorgnome radio pulse detection

`gnu-radio-host.py` is used to communicate to `leberrigan/sensorgnome-control/src/grh.js`.
It initializes new radios and sends data to the main SG control process.

`gr_detect_pulses.py` is the main Gnu Radio block for detecting pulses.

`gr_airspy.py`, `gr_airspyhf.py`, and `gr_rtlsdr.py` are the flow graphs for each respective dongle.

All files with the suffix ".grc" are GnuRadio Companion flow graph configurations for testing with a GUI.

### Detection algorithm (`detect_pulse_2.py`)

Pulses are detected using a matched-filter (rising-edge template) cross-correlated via FFT against
the signal magnitude — O(N + N log N) per chunk.  After each detection, the dominant frequency
component is estimated via phase regression on the pulse body and subtracted from the IQ signal
(tone subtraction).  This is the primary mechanism for separating overlapping pulses.

**Tone subtraction note**: subtraction quality depends on the sample rate.  At 192 kHz there are
~38 samples of clean plateau for phase regression; at 48 kHz (after 4x decimation) there are ~32
samples (~2 cycles at the lowest tracked frequency).  Below ~32 samples the phase estimate becomes
unreliable and subtraction silently declines.  The detector still achieves full detection accuracy
without subtraction — it is only needed for overlap disambiguation.

### High-performance vs low-performance mode

`detect_pulse_2.Pulse_Detector` and `gr_detect_pulses.blk` both accept a `high_perf` flag
(default `True`).

| Mode | What it does | CPU cost | When to use |
|------|-------------|----------|-------------|
| `high_perf=True` | Matched filter + FFT + tone subtraction | ~0.41x realtime per instance at 48 kHz on RPi 3B | 2–3 simultaneous dongles |
| `high_perf=False` | Matched filter only (no FFT, no subtraction) | ~5–10x less CPU | 10+ simultaneous dongles |

To activate low-performance mode pass `--low_perf` on the CLI when starting a flow graph:

```
python3 gr_rtlsdr.py --low_perf ...
python3 gr_airspyhf.py --low_perf ...
```

sensorgnome-control can include `--low_perf` in the `open` command's `additional_args` field
to switch all devices to low-performance mode when the dongle count exceeds a threshold.

In low-performance mode the frequency offset field in CSV output is always 0; all other fields
(timestamp, plateau dB, SNR dB, duration ms) remain fully populated.

### Testing

Use `read-raw-iq.py` for reading raw IQ data and producing plots for inspection of detector performance.

Run `demo.py` for a self-contained benchmark: accuracy (94/100), speed, RPi 3B capacity estimate,
sample detection output, and old-vs-new comparison.

Run `airspy_time_domain_2.py` to produce three HTML visualisation files from `test_signal.iq`:
- `edge_templates.html` — matched-filter template shape
- `time_domain_analysis.html` — I/Q, magnitude (original vs after subtraction), correlations, detected edges vs ground truth
- `timing_error_analysis.html` — timing error histogram with Gaussian fit and per-pulse scatter

### Gnu Radio Companion Flow Graphs

- All flow graphs (with extension ".grc") are set up to use `detect_pulse_2.py`, but it may be necessary to point the python block to the directory by adding to the start of the python block:

    ```
    import sys
    sys.path.insert(0, "C:/PATH/TO/FOLDER/")
    ```
