#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: AirSpy pulse detector
# Author: Lucas Berrigan
# GNU Radio version: 3.10.12.0


from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
import json

from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import gr_detect_pulses as detect_pulses  # embedded python block
import osmosdr
import time
import threading

class airspy_detect_pulse(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "AirSpy pulse detector", catch_exceptions=True)
        
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.args = self.get_args()

        self.samp_rate = samp_rate = float(self.args.samp_rate)
        self.freq = freq = self.args.freq
        self.freq_offset = freq_offset = float(4e3)
        self.filter_cutoff_freq = filter_cutoff_freq = 12e3
        self.filter_transition_width = filter_transition_width = 8e3
        self.device = device = self.args.device
        self.verbose = verbose = self.args.verbose
        self.port = port = self.args.port
        self.decimation_factor = decimation_factor = int( samp_rate / self.args.target_rate )
        self.additional_args = additional_args = self.args.additional_args
        self.pulse_len_ms = 2.5
        self.max_pulse_samples = int((self.pulse_len_ms / 1000) * (self.samp_rate / self.decimation_factor))
        self.gain = json.loads(self.args.gain)
        self.gain_rf = float(self.gain.get('rf', 15))
        self.gain_bb = float(self.gain.get('bb', 20))
        self.gain_if = float(self.gain.get('if', 0))

        ##################################################
        # Blocks
        ##################################################
        
        self.osmosdr_source = osmosdr.source(
            args=f"airspy={device},bias=0,{additional_args}"
        )
        self.osmosdr_source.set_time_unknown_pps(osmosdr.time_spec_t())
        self.osmosdr_source.set_sample_rate(samp_rate)
        self.osmosdr_source.set_center_freq((freq - freq_offset), 0)
        self.osmosdr_source.set_freq_corr(0, 0)
        self.osmosdr_source.set_dc_offset_mode(0, 0)
        self.osmosdr_source.set_iq_balance_mode(0, 0)
        self.osmosdr_source.set_gain_mode(False, 0)
        self.osmosdr_source.set_gain(self.gain_rf, 0)
        self.osmosdr_source.set_if_gain(self.gain_if, 0)
        self.osmosdr_source.set_bb_gain(self.gain_bb, 0)
        self.osmosdr_source.set_antenna('', 0)
        self.osmosdr_source.set_bandwidth(0, 0)
        self.freq_xlating_fir_filter = filter.freq_xlating_fir_filter_ccc(
            self.decimation_factor, 
            firdes.low_pass(
                1.0, 
                samp_rate, 
                filter_cutoff_freq, 
                filter_transition_width, 
                window.WIN_HAMMING
            ), 
            self.freq_offset, 
            self.samp_rate
        )
        self.detect_pulses = detect_pulses.blk(
            output_type="stream",
            verbose=self.verbose,
            port=port,
            samp_rate=samp_rate / decimation_factor,
            min_snr_db=6,
            debounce_samples=10,
            pulse_len_ms=2.5
        )

        ##################################################
        # Connections
        ##################################################
        self.connect((self.osmosdr_source, 0), (self.freq_xlating_fir_filter, 0))
        self.connect((self.freq_xlating_fir_filter, 0), (self.detect_pulses, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.detect_pulses.samp_rate = self.samp_rate / self.decimation_factor
        self.freq_xlating_fir_filter.set_taps(firdes.low_pass(1.0, self.samp_rate, filter_cutoff_freq, filter_transition_width, window.WIN_HAMMING))
        self.osmosdr_source.set_sample_rate(self.samp_rate)

    def get_freq_offset(self):
        return self.freq_offset

    def set_freq_offset(self, freq_offset):
        self.freq_offset = freq_offset
        self.freq_xlating_fir_filter.set_center_freq(self.freq_offset)
        self.osmosdr_source.set_center_freq((self.freq - self.freq_offset), 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.osmosdr_source.set_center_freq((self.freq - self.freq_offset), 0)
        return "success"

    def get_fft_size(self):
        return self.fft_size

    def set_fft_size(self, fft_size):
        self.fft_size = fft_size

    def get_decimation_factor(self):
        return self.decimation_factor

    def set_decimation_factor(self, decimation_factor):
        self.decimation_factor = decimation_factor
        self.detect_pulses.samp_rate = self.samp_rate / self.decimation_factor

    def set_freq(self, freq):
        self.freq = float(freq)
        self.osmosdr_source.set_center_freq(self.freq - self.freq_offset, 0)
        return "success"

    def set_rf_gain(self, gain):
        self.gain_rf = float(gain)
        self.osmosdr_source.set_gain(self.gain_rf, 0)
        return "success"

    def set_sensitivity_gain(self, gain):
        self.osmosdr_source.set_gain(float(gain), "SENSITIVITY", 0)
        return "success"

    def set_agc(self, enabled):
        self.osmosdr_source.set_gain_mode(bool(int(enabled)), 0)
        return "success"

    def set_bias_tee(self, enabled):
        # osmosdr has no runtime bias-tee API; value is set via device args at startup
        return "success"

    def read_stdin(self):
        for line in sys.stdin:
            parts = line.strip().split()
            if not parts: continue
            action = parts[0]
            args = parts[1:]
            if action in ("set_freq", "set_rf_gain", "set_sensitivity_gain", "set_agc", "set_bias_tee") and args:
                try:
                    response = getattr(self, action)(*args)
                    if response:
                        print(response, flush=True)
                except Exception as e:
                    print(f"error: {action} {args}: {e}", flush=True)

    # Set up command-line argument parsing
    def get_args(self):
        parser = ArgumentParser(description='GNU Radio flow graph')
        parser.add_argument('-p', '--port', required=True, help='Device port', type = int)
        parser.add_argument('-d', '--device', required=True, help='Device ID', type = str)
        parser.add_argument('-tr','--target_rate', help='Target sample rate', default = 48e3, type = int)
        parser.add_argument('-s', '--samp_rate', help='Hardware sample rate', default = 3e6, type = int)
        parser.add_argument('-f', '--freq', help='Frequency', default = 166376000, type = float)
        parser.add_argument('-v', '--verbose', help='Print messages', default = False, action="store_true")
        parser.add_argument('-a', '--additional_args', help='Arguments to pass on to osmosdr on init', default = "sensitivity=21", type = str)
        parser.add_argument('-g', '--gain', help='Gain as JSON, e.g. \'{"rf":15,"bb":20,"if":0}\'', default = '{}', type = str)
        return parser.parse_args()



def main(top_block_cls=airspy_detect_pulse, options=None):

    tb = top_block_cls()
    tb.start()
    tb.flowgraph_started.set()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    threading.Thread(target=tb.read_stdin, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sig_handler()


if __name__ == '__main__':
    main()
