## Gnu Radio host, flow graphs, and blocks for Sensorgnome radio pulse detection

`gnu-radio-host.py` is used to communicate to `leberrigan/sensorgnome-control/src/grh.js`.
It initializes new radios and sends data to the main SG control process.

`gr_detect_pulses.py` is the main Gnu Radio block for detecting pulses.

`gr_airspy.py`, `gr_airspyhf.py`, and `gr_rtlsdr.py` are the flow graphs for each respective dongle.