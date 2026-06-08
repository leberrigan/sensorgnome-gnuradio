#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: FUNcube Dongle Pro+ pulse detector
# Author: Lucas Berrigan
# GNU Radio version: 3.10.12.0
#
# The FCD Pro+ presents as a USB stereo audio device (192 kHz IQ):
#   left channel = I,  right channel = Q
# RF tuning is done via the `fcd` utility (USB HID) rather than osmosdr.
#
# --device argument is a compound string "BUS:USBDEV:ALSA_CARD"
# e.g. "1:22:2" means USB path 1:22 (for `fcd -p`) and ALSA card 2 (for audio.source).

from gnuradio import audio, blocks, filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
import json
import subprocess

from argparse import ArgumentParser
from gnuradio import eng_notation
import gr_detect_pulses as detect_pulses
import time
import threading


class funcubepp_detect_pulse(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "FUNcube Pro+ pulse detector", catch_exceptions=True)

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
        self.verbose = verbose = self.args.verbose
        self.port = port = self.args.port
        self.decimation_factor = decimation_factor = int(samp_rate / self.args.target_rate)
        self.additional_args = additional_args = self.args.additional_args
        self.gain = json.loads(self.args.gain)
        self.gain_lna = float(self.gain.get('lna', 20))
        self.gain_if = float(self.gain.get('if', 20))

        # Parse compound device ID: "BUS:USBDEV:ALSA_CARD" e.g. "1:22:2"
        parts = self.args.device.rsplit(":", 1)
        self.usb_path = parts[0]    # e.g. "1:22"  — used by `fcd -p` for RF tuning
        self.alsa_card = parts[1]   # e.g. "2"     — used by audio.source

        ##################################################
        # Set initial RF tuning frequency
        ##################################################
        self._tune_fcd(int(freq - freq_offset))

        ##################################################
        # Blocks
        ##################################################

        # FCD Pro+ outputs stereo IQ audio: left=I, right=Q
        self.audio_source = audio.source(int(samp_rate), f"plughw:{self.alsa_card}", True)

        # Combine I and Q float streams into a single complex stream
        self.float_to_complex = blocks.float_to_complex(1)

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
            pulse_len_ms=2.5,
        )

        ##################################################
        # Connections
        # Audio source: port 0 = left (I), port 1 = right (Q)
        ##################################################
        self.connect((self.audio_source, 0), (self.float_to_complex, 0))
        self.connect((self.audio_source, 1), (self.float_to_complex, 1))
        self.connect((self.float_to_complex, 0), (self.freq_xlating_fir_filter, 0))
        self.connect((self.freq_xlating_fir_filter, 0), (self.detect_pulses, 0))

    def _tune_fcd(self, freq_hz):
        try:
            result = subprocess.run(
                ["/usr/bin/fcd", "-p", self.usb_path, "-q", "-s", str(int(freq_hz))],
                capture_output=True, timeout=5
            )
            if result.returncode != 0:
                print(f"fcd tuning warning: {result.stderr.decode().strip()}", flush=True)
        except Exception as e:
            print(f"fcd tuning error: {e}", flush=True)

    def set_freq(self, freq):
        self.freq = float(freq)
        self._tune_fcd(int(self.freq - self.freq_offset))
        self.freq_xlating_fir_filter.set_center_freq(self.freq_offset)
        return "success"

    def set_lna_gain(self, gain):
        self.gain_lna = float(gain)
        # LNA gain on FCD Pro+ is not directly accessible at runtime via audio source.
        # Would require a separate fcd parameter command; not currently supported.
        return "success"

    def read_stdin(self):
        for line in sys.stdin:
            parts = line.strip().split()
            if not parts: continue
            action = parts[0]
            args = parts[1:]
            if action in ("set_freq", "set_lna_gain") and args:
                try:
                    response = getattr(self, action)(*args)
                    if response:
                        print(response, flush=True)
                except Exception as e:
                    print(f"error: {action} {args}: {e}", flush=True)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.detect_pulses.samp_rate = self.samp_rate / self.decimation_factor
        self.freq_xlating_fir_filter.set_taps(
            firdes.low_pass(1.0, self.samp_rate, self.filter_cutoff_freq,
                            self.filter_transition_width, window.WIN_HAMMING)
        )

    def get_freq(self):
        return self.freq

    def get_decimation_factor(self):
        return self.decimation_factor

    def get_args(self):
        parser = ArgumentParser(description='GNU Radio flow graph — FUNcube Dongle Pro+')
        parser.add_argument('-p', '--port', required=True, help='Device port', type=int)
        parser.add_argument('-d', '--device', required=True,
                            help='Compound device ID "BUS:USBDEV:ALSA_CARD", e.g. "1:22:2"',
                            type=str)
        parser.add_argument('-tr', '--target_rate', help='Target sample rate', default=192000, type=int)
        parser.add_argument('-s', '--samp_rate', help='Hardware sample rate', default=192000, type=int)
        parser.add_argument('-f', '--freq', help='Centre frequency in Hz', default=166376000, type=float)
        parser.add_argument('-v', '--verbose', help='Print messages', default=False, action='store_true')
        parser.add_argument('-a', '--additional_args', help='Reserved for future use', default='', type=str)
        parser.add_argument('-g', '--gain', help='Gain as JSON, e.g. \'{"lna":20,"if":20}\'',
                            default='{"lna":20,"if":20}', type=str)
        parser.add_argument('--low_perf', help='Disable FFT/subtraction for lower CPU use',
                            default=False, action='store_true')
        return parser.parse_args()


def main(top_block_cls=funcubepp_detect_pulse, options=None):

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
