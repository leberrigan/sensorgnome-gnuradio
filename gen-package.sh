#! /bin/bash -e
DESTDIR=build-temp
PKGNAME=sensorgnome-gnuradio

# Clean previous build
rm -rf "$DESTDIR"
mkdir -p "$DESTDIR"

# ── SoapyAirspyHF .so (ARM native build via Docker + qemu) ───────────────────
# soapysdr-module-airspyhf is not in Debian Bookworm apt repos so we build from
# source here and bundle the .so directly in the package.
SOAPY_OUT="$(mktemp -d)"
docker run --privileged --rm tonistiigi/binfmt --install arm
docker run --rm \
    --platform linux/arm/v7 \
    -v "${SOAPY_OUT}:/output" \
    arm32v7/debian:bookworm \
    bash -exc "
        apt-get update -qq
        apt-get install -y --no-install-recommends \
            cmake build-essential git ca-certificates \
            libsoapysdr-dev libairspyhf-dev
        git clone --depth 1 https://github.com/pothosware/SoapyAirspyHF.git /tmp/sa
        cmake -S /tmp/sa -B /tmp/sa/build \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/usr
        make -C /tmp/sa/build -j2
        DESTDIR=/output make -C /tmp/sa/build install
        chown -R $(id -u):$(id -g) /output
    "
# Merge the installed files (just the .so tree) into the package staging dir
cp -a "${SOAPY_OUT}/." "${DESTDIR}/"
rm -rf "${SOAPY_OUT}"

# ── Executables → /usr/bin/ ──────────────────────────────────────────────────
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
