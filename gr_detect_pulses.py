# GRC Python Block body (sync_block)
# Inputs: 1 (complex64 raw i/q)
# Outputs: 0
# Lucas Berrigan 2025
#
# Bridges the GNU Radio flow graph to the pulse detector.
# Computes magnitude from complex IQ internally so the flow graph
# only needs a single complex input port.

import numpy as np
import detect_pulse_3 as pd

from gnuradio import gr

class blk(gr.sync_block):

    def __init__(self, output_type="file", verbose=True, filename="output.csv", port=0, samp_rate=250000, min_snr_db=6, debounce_samples: int=10, pulse_len_ms: float=2.5, high_perf: bool=True):
        gr.sync_block.__init__(self,    
            name="Pulse detector",
            in_sig=[np.complex64],
            out_sig=[])
        self.pulse_detector = pd.Pulse_Detector(output_type, verbose, filename, port, samp_rate, min_snr_db, debounce_samples, pulse_len_ms, high_perf=high_perf)

    def work(self, input_items, output_items):
        iq = input_items[0]
        return self.pulse_detector.detect([iq])

    def __del__(self):
        pass
