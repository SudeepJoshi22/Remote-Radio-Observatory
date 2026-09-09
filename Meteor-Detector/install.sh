#!/usr/bin/env bash
# Sets up the virtualenv for the Remote Radio Observatory.
#
#   ./install.sh
#   source venv/bin/activate      <- must be run in YOUR shell, not this script
#
set -euo pipefail
cd "$(dirname "$0")"

sudo apt install -y rtl-sdr librtlsdr-dev

python3 -m venv venv
./venv/bin/pip install --upgrade pip setuptools
./venv/bin/pip install -r ../requirements.txt

cat <<'MSG'

Done. Activate the environment in your own shell:

    source venv/bin/activate

Then verify the DSP before touching hardware:

    cd acquisition
    python3 rf_check.py --selftest
    python3 test_pipeline.py
MSG
