#!/usr/bin/env bash
#
# Sets up the Remote Radio Observatory on Debian/Raspberry Pi OS.
#
#   ./install.sh
#   source venv/bin/activate      <- must be run in YOUR shell, not this script
#
# Installs librtlsdr and the Python environment, and -- importantly -- stops the
# kernel from claiming the dongle as a TV tuner. This is idempotent; re-running
# it is harmless.
#
# Windows users: this script does not apply. Install the RTL-SDR DLLs and use
# Zadig to bind WinUSB to the dongle. See acquisition/README.md.

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v apt >/dev/null 2>&1; then
    echo "This installer targets Debian / Raspberry Pi OS (apt not found)."
    echo "Install librtlsdr and its Python bindings by hand, then re-run the"
    echo "verification step at the end of this script."
    exit 1
fi

IS_WSL=0
if grep -qi "microsoft\|WSL" /proc/version 2>/dev/null; then
    IS_WSL=1
    cat <<'WSLMSG'
==> WSL detected

WSL2 has no direct USB access. The dongle must be attached from the Windows
side with usbipd-win before anything here can see it:

    # once, in Windows PowerShell
    winget install usbipd

    # each time, in PowerShell AS ADMINISTRATOR
    usbipd list                          # find the dongle's BUSID
    usbipd bind   --busid <BUSID>        # once per device
    usbipd attach --wsl --busid <BUSID>  # after every reboot / replug

Then `lsusb` inside WSL should list it. Zadig is NOT needed for this route --
usbipd forwards the raw device and the Linux side owns the driver.

Two WSL caveats worth knowing:
  - udev may not run without systemd, so device permissions can need sudo.
  - USB-over-IP adds latency. If fm_observe.py reports dropped blocks, lower
    --sample-rate to 250e3 or run on real hardware.

Continuing with the Linux setup.

WSLMSG
fi

echo "==> Installing librtlsdr and tools"
sudo apt update
sudo apt install -y rtl-sdr librtlsdr-dev python3-venv python3-pip

# --------------------------------------------------------------------------
# Stop the DVB-T driver claiming the dongle.
#
# The kernel sees an RTL2832U and loads dvb_usb_rtl28xxu, treating it as a
# television tuner. librtlsdr then cannot claim the USB interface and every
# tool fails with "usb_claim_interface error -6" or "Failed to open rtlsdr
# device #0". This is the most common RTL-SDR problem on Linux and nothing
# else in the setup will work until it is fixed.
# --------------------------------------------------------------------------
BLACKLIST=/etc/modprobe.d/blacklist-rtlsdr.conf
echo "==> Blacklisting the DVB-T kernel driver ($BLACKLIST)"
echo "    NOTE: this stops these dongles working as actual DVB-T television"
echo "    receivers. To use one for TV again, delete that file and reboot."
echo "    It does NOT matter which dongle you use as an SDR -- a generic"
echo "    DVB-T stick and a purpose-built RTL-SDR both need this."
sudo tee "$BLACKLIST" >/dev/null <<'BL'
# Keep the kernel's DVB-T driver away from RTL-SDR dongles so librtlsdr can
# claim them. Written by Remote-Radio-Observatory/Meteor-Detector/install.sh
blacklist dvb_usb_rtl28xxu
blacklist rtl2832
blacklist rtl2830
BL

RELOAD_NEEDED=0
for mod in dvb_usb_rtl28xxu rtl2832 rtl2830; do
    if lsmod | grep -q "^${mod} "; then
        echo "    unloading $mod"
        sudo modprobe -r "$mod" 2>/dev/null || RELOAD_NEEDED=1
    fi
done

# --------------------------------------------------------------------------
# Non-root access to the device.
# --------------------------------------------------------------------------
echo "==> Checking udev rules"
if ls /lib/udev/rules.d/*librtlsdr* /etc/udev/rules.d/*rtlsdr* >/dev/null 2>&1; then
    echo "    packaged udev rules already present"
else
    echo "    installing /etc/udev/rules.d/60-rtlsdr.rules"
    sudo tee /etc/udev/rules.d/60-rtlsdr.rules >/dev/null <<'UDEV'
# RTL-SDR dongles, readable without root. Written by install.sh
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2832", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="0ccd", ATTRS{idProduct}=="00a9", GROUP="plugdev", MODE="0666"
SUBSYSTEM=="usb", ATTRS{idVendor}=="1f4d", ATTRS{idProduct}=="b803", GROUP="plugdev", MODE="0666"
UDEV
fi
sudo udevadm control --reload-rules 2>/dev/null || true
sudo udevadm trigger 2>/dev/null || true

if getent group plugdev >/dev/null 2>&1; then
    if ! id -nG "$USER" | tr ' ' '\n' | grep -qx plugdev; then
        echo "==> Adding $USER to the plugdev group"
        sudo usermod -aG plugdev "$USER"
        echo "    log out and back in for this to take effect"
    fi
fi

# --------------------------------------------------------------------------
echo "==> Creating the Python environment"
python3 -m venv venv
./venv/bin/pip install --quiet --upgrade pip setuptools
./venv/bin/pip install --quiet -r ../requirements.txt

# --------------------------------------------------------------------------
echo "==> Verifying the DSP (no hardware needed)"
( cd acquisition && ../venv/bin/python rf_check.py --selftest >/dev/null \
  && echo "    self test PASSED" ) || {
    echo "    self test FAILED -- stop here and investigate"; exit 1; }

echo
echo "==> Checking for a dongle"
if command -v rtl_test >/dev/null 2>&1; then
    if timeout 6 rtl_test -t 2>&1 | grep -qi "Found 1\|Found [0-9]* device"; then
        echo "    dongle detected and openable"
    else
        echo "    no working dongle detected right now."
        echo "    If one IS plugged in, UNPLUG AND REPLUG IT -- the DVB-T driver"
        echo "    was blacklisted in this run and the device must be re-enumerated."
        [ "$RELOAD_NEEDED" = "1" ] && echo "    A reboot may be required."
    fi
fi

cat <<'MSG'

Done. Activate the environment in your own shell:

    source venv/bin/activate

Then, before touching hardware:

    cd acquisition
    python3 rf_check.py --selftest
    python3 test_pipeline.py

With more than one dongle attached, check which index is which:

    python3 rf_check.py --list-devices

Then the RF chain verdict (add --no-lna if there is no preamp,
and -D <idx> to pick a dongle):

    python3 rf_check.py --floor-test --no-lna -g 49.6 -D 0

If any tool reports "Failed to open rtlsdr device #0" or
"usb_claim_interface error -6", the kernel still has the dongle:

    lsmod | grep dvb        # should print nothing
    sudo modprobe -r dvb_usb_rtl28xxu
    # then unplug and replug the dongle
MSG
