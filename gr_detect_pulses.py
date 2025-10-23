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
import numpy as np
import detect_pulse as pd

from gnuradio import gr

class blk(gr.sync_block):

    def __init__(self, output_type="file", verbose=True, filename="output.csv", port = 0, samp_rate=250000, min_snr_db=6, debounce_samples: int=10, pulse_len_ms: float=2.5):
        gr.sync_block.__init__(self,
            name="Pulse detector",
            in_sig=[np.float32, np.complex64],  # magnitude, raw iq
            out_sig=[])
        self.pulse_detector = pd.Pulse_Detector(output_type, verbose, filename, port, samp_rate, min_snr_db, debounce_samples, pulse_len_ms)
        
    def work(self, input_items, output_items):
        return self.pulse_detector.detect(input_items)

    def __del__(self):
        pass