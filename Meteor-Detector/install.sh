sudo apt install rtl-sdr
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install "numpy>=2.2.3" "scipy>=1.15.1" "matplotlib>=3.10.0" "pandas>=2.2.3" "pyrtlsdr>=0.3.0"
pip install setuptools
