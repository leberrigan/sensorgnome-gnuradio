#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: AirSpy pulse detector
# Author: Lucas Berrigan
# GNU Radio version: 3.10.12.0


from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal

from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import gr_detect_pulses as detect_pulses  # embedded python block
from gnuradio import soapy
import time
import threading
import json

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

        self.gain = json.loads( self.args.gain )
        self.gain_rf = self.gain['rf'] if 'rf' in self.gain else -24
        self.gain_lna = 6 if 'lna' in self.gain and self.gain['lna'] == 6 else 0

        ##################################################
        # Blocks
        ##################################################
        
        self.soapy_airspyhf_source = None
        dev = 'driver=airspyhf'
        serial = self.device
        stream_args = additional_args
        tune_args = ['']
        settings = ['']

        self.soapy_airspyhf_source = soapy.source(
            dev,
            "fc32", 
            1, 
            f"serial={serial}",
            stream_args, 
            tune_args, 
            settings
            )
        self.soapy_airspyhf_source.set_sample_rate(0, samp_rate)
        self.soapy_airspyhf_source.set_gain_mode(0, False)
        self.soapy_airspyhf_source.set_frequency(0, (freq-freq_offset))
        self.soapy_airspyhf_source.set_frequency_correction(0, 0)
        self.soapy_airspyhf_source.set_gain(0, 'RF', min(max((self.gain_rf), -48.0), 0.0))
        self.soapy_airspyhf_source.set_gain(0, 'LNA', self.gain_rf)

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
        self.blocks_moving_average = blocks.moving_average_ff( 5, 1/5, self.max_pulse_samples * 10, 1)
        self.blocks_complex_to_mag_squared = blocks.complex_to_mag_squared(1)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_complex_to_mag_squared, 0), (self.blocks_moving_average, 0))
        self.connect((self.blocks_moving_average, 0), (self.detect_pulses, 0))
        self.connect((self.freq_xlating_fir_filter, 0), (self.blocks_complex_to_mag_squared, 0))
        self.connect((self.freq_xlating_fir_filter, 0), (self.detect_pulses, 1))
        self.connect((self.soapy_airspyhf_source, 0), (self.freq_xlating_fir_filter, 0))


    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.detect_pulses.samp_rate = self.samp_rate / self.decimation_factor
        self.freq_xlating_fir_filter.set_taps(firdes.low_pass(1.0, self.samp_rate, filter_cutoff_freq, filter_transition_width, window.WIN_HAMMING))
        self.soapy_airspyhf_source.set_sample_rate(self.samp_rate)

    def get_freq_offset(self):
        return self.freq_offset

    def set_freq_offset(self, freq_offset):
        self.freq_offset = freq_offset
        self.freq_xlating_fir_filter.set_center_freq(self.freq_offset)
        self.soapy_airspyhf_source.set_frequency((self.freq - self.freq_offset), 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.soapy_airspyhf_source.set_frequency((self.freq - self.freq_offset), 0)
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



    # Set up command-line argument parsing
    def get_args(self):
        parser = ArgumentParser(description='GNU Radio flow graph')
        parser.add_argument('-p', '--port', required=True, help='Device port', type = int)
        parser.add_argument('-d', '--device', required=True, help='Device ID', type = str)
        parser.add_argument('-tr','--target_rate', help='Target sample rate', default = 48e3, type = int)
        parser.add_argument('-s', '--samp_rate', help='Hardware sample rate', default = 3e6, type = int)
        parser.add_argument('-f', '--freq', help='Frequency', default = 166376000, type = float)
        parser.add_argument('-v', '--verbose', help='Print messages', default = False, action="store_true")
        parser.add_argument('-a', '--additional_args', help='Arguments to pass on to soapysdr on init', default = "", type = str)
        parser.add_argument('-g', '--gain', help='Gain', default = "{\"rf\":-24,\"lna\":6}", type = str)
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

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sig_handler()


if __name__ == '__main__':
    main()
