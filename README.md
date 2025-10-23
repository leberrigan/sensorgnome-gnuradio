## Gnu Radio host, flow graphs, and blocks for Sensorgnome radio pulse detection

`gnu-radio-host.py` is used to communicate to `leberrigan/sensorgnome-control/src/grh.js`.
It initializes new radios and sends data to the main SG control process.

`gr_detect_pulses.py` is the main Gnu Radio block for detecting pulses.

`gr_airspy.py`, `gr_airspyhf.py`, and `gr_rtlsdr.py` are the flow graphs for each respective dongle.

All files with the suffix ".grc" are GnuRadio Companion flow graph configurations for testing with a GUI.

### Testing

Use `read-raw-iq.py` for reading raw IQ data and producing plots for inspection of `detect_pulse.py` performance.

### Gnu Radio Companion Flow Graphs

- All flow graphs (with extension ".grc") are set up to use `detect_pulse.py`, but it may be necessary to point the python block to the directory by adding to the start of the python block:

    ```
    import sys
    sys.path.insert(0, "C:/PATH/TO/FOLDER/")
    ```
