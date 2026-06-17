#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: AirSpy HF+ pulse detector
# Author: Lucas Berrigan
# GNU Radio version: 3.10.12.0

from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
import json
import time
import threading

from argparse import ArgumentParser
import gr_detect_pulses as detect_pulses
from gnuradio import soapy


class airspyhf_detect_pulse(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "AirSpy HF+ pulse detector", catch_exceptions=True)

        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.args = self.get_args()

        self.samp_rate      = samp_rate      = float(self.args.samp_rate)
        self.freq           = freq           = self.args.freq
        self.freq_offset    = freq_offset    = float(self.args.freq_offset)
        self.filter_cutoff_freq      = filter_cutoff_freq      = 12e3
        self.filter_transition_width = filter_transition_width = 24e3
        self.device         = device         = self.args.device
        self.verbose        = verbose        = self.args.verbose
        self.port           = port           = self.args.port
        self.decimation_factor = decimation_factor = int(samp_rate / self.args.target_rate)
        self.additional_args = additional_args = self.args.additional_args

        self.gain     = json.loads(self.args.gain)
        # RF gain: 0 dB = max sensitivity, -48 dB = max attenuation
        self.gain_rf  = float(self.gain.get('rf',  -24))
        self.gain_lna = 6 if self.gain.get('lna', 6) == 6 else 0
        self.agc      = bool( self.gain.get('agc', False))

        ##################################################
        # Blocks
        ##################################################
        # Strip udevadm prefix (e.g. "AIRSPYHF_SN:") — SoapySDR wants only the hex serial
        serial = device
        if ':' in serial:
            serial = serial.split(':', 1)[1]

        self.soapy_airspyhf_source = soapy.source(
            'driver=airspyhf',
            'fc32',
            1,
            f'serial={serial}',
            additional_args,
            [''],
            [''],
        )
        self.soapy_airspyhf_source.set_sample_rate(0, samp_rate)
        self.soapy_airspyhf_source.set_gain_mode(0, self.agc)
        self.soapy_airspyhf_source.set_frequency(0, freq - freq_offset)
        self.soapy_airspyhf_source.set_frequency_correction(0, 0)
        if not self.agc:
            self.soapy_airspyhf_source.set_gain(0, 'RF',  min(max(self.gain_rf,  -48.0), 0.0))
            self.soapy_airspyhf_source.set_gain(0, 'LNA', self.gain_lna)

        self.freq_xlating_fir_filter = filter.freq_xlating_fir_filter_ccc(
            decimation_factor,
            firdes.low_pass(
                1.0,
                samp_rate,
                filter_cutoff_freq,
                filter_transition_width,
                window.WIN_HAMMING,
            ),
            freq_offset,
            samp_rate,
        )

        self.detect_pulses = detect_pulses.blk(
            output_type='stream',
            verbose=verbose,
            port=port,
            samp_rate=samp_rate / decimation_factor,
            min_snr_db=6,
            debounce_samples=10,
            pulse_len_ms=2.5,
            high_perf=not self.args.low_perf,
        )

        ##################################################
        # Connections
        ##################################################
        self.connect((self.soapy_airspyhf_source,    0), (self.freq_xlating_fir_filter, 0))
        self.connect((self.freq_xlating_fir_filter,  0), (self.detect_pulses,           0))

    # ------------------------------------------------------------------
    # Runtime setters (called from read_stdin)
    # ------------------------------------------------------------------

    def set_freq(self, freq):
        self.freq = float(freq)
        self.soapy_airspyhf_source.set_frequency(0, self.freq - self.freq_offset)
        return 'success'

    def set_rf_gain(self, gain):
        self.gain_rf = min(max(float(gain), -48.0), 0.0)
        self.soapy_airspyhf_source.set_gain(0, 'RF', self.gain_rf)
        return 'success'

    def set_lna_gain(self, gain):
        self.gain_lna = 6 if int(gain) == 6 else 0
        self.soapy_airspyhf_source.set_gain(0, 'LNA', self.gain_lna)
        return 'success'

    def set_sensitivity_gain(self, gain):
        # Values are in dB (0 = max sensitivity / no attenuation, -48 = max attenuation).
        return self.set_rf_gain(float(gain))

    # ------------------------------------------------------------------
    # stdin command loop
    # ------------------------------------------------------------------

    def read_stdin(self):
        for line in sys.stdin:
            parts = line.strip().split()
            if not parts:
                continue
            action, args = parts[0], parts[1:]
            if action in ('set_freq', 'set_rf_gain', 'set_lna_gain', 'set_sensitivity_gain') and args:
                try:
                    response = getattr(self, action)(*args)
                    if response:
                        print(response, file=sys.stderr, flush=True)
                except Exception as e:
                    print(f'error: {action} {args}: {e}', flush=True)

    # ------------------------------------------------------------------
    # Argument parser
    # ------------------------------------------------------------------

    def get_args(self):
        parser = ArgumentParser(description='GNU Radio flow graph — AirSpy HF+')
        parser.add_argument('-p', '--port',          required=True,                       type=int)
        parser.add_argument('-d', '--device',        required=True,                       type=str)
        parser.add_argument('-tr','--target_rate',   default=48000,                       type=int)
        parser.add_argument('-s', '--samp_rate',     default=192000,                      type=int)
        parser.add_argument('-f', '--freq',          default=166376000,                   type=float)
        parser.add_argument('-fo','--freq_offset',   default=4000,                        type=float,
                            help='Hz the radio is tuned below --freq (the signal is digitally '
                                 'shifted back, so reported offsets are unchanged). Increase it '
                                 '(e.g. 8000-10000) to move tags further from the DC spike / '
                                 'near-DC birdies. Keep below the filter cutoff (12 kHz).')
        parser.add_argument('-v', '--verbose',       default=False, action='store_true')
        parser.add_argument('-a', '--additional_args', default='',                        type=str)
        parser.add_argument('-g', '--gain',
                            default='{"rf":-24,"lna":6}',                                type=str,
                            help='Gain as JSON: {"rf":-24,"lna":6} (rf: 0=max, -48=min) or {"agc":true}')
        parser.add_argument('--low_perf', default=False, action='store_true')
        return parser.parse_args()


def main(top_block_cls=airspyhf_detect_pulse):

    tb = top_block_cls()
    tb.start()
    tb.flowgraph_started.set()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT,  sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    threading.Thread(target=tb.read_stdin, daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sig_handler()


if __name__ == '__main__':
    main()
