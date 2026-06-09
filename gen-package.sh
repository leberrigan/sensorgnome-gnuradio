#! /bin/bash -e
DESTDIR=build-temp
PKGNAME=sensorgnome-gnuradio

# Clean previous build
rm -rf "$DESTDIR"
mkdir -p "$DESTDIR"

# ── Executables → /usr/bin/ ──────────────────────────────────────────────────
# Note: SoapyAirspyHF (.so) is built from source directly in sg-armv7-rpi-bookworm.pifile
# and does not need to be bundled in this package.
install -d "$DESTDIR/usr/bin"
install -m 755 gnu-radio-host.py  "$DESTDIR/usr/bin/"
install -m 755 gr_airspy.py        "$DESTDIR/usr/bin/"
install -m 755 gr_airspyhf.py      "$DESTDIR/usr/bin/"
install -m 755 gr_rtlsdr.py        "$DESTDIR/usr/bin/"
install -m 755 gr_funcubepp.py     "$DESTDIR/usr/bin/"

# ── Python library modules → /usr/lib/python3/dist-packages/ ─────────────────
# These are imported by the flow graphs; placing them here puts them on
# the system Python path so `import gr_detect_pulses` etc. work regardless
# of the working directory.
PYLIB="$DESTDIR/usr/lib/python3/dist-packages"
install -d "$PYLIB"
install -m 644 gr_detect_pulses.py     "$PYLIB/"
install -m 644 detect_pulse_2.py       "$PYLIB/"
install -m 644 detect_pulse_overlap.py "$PYLIB/"
install -m 644 detect_pulse.py         "$PYLIB/"

# ── DEBIAN control files ──────────────────────────────────────────────────────
cp -r DEBIAN "$DESTDIR"
chmod 0755 "$DESTDIR"/DEBIAN/post* 2>/dev/null || true
chmod 0755 "$DESTDIR"/DEBIAN/pre*  2>/dev/null || true

# Set package version: YYYY.DDD (same convention as other sensorgnome packages)
sed -e "/^Version/s/:.*/: $(TZ=PST8PDT date +%Y.%j)/" \
    -i "$DESTDIR/DEBIAN/control"

# ── Build ─────────────────────────────────────────────────────────────────────
mkdir -p packages
dpkg-deb -Zxz --build "$DESTDIR" packages

ls -lh packages
