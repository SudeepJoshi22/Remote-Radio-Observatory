# Scans the frequency band over 10s and outputs the bins per frequency point in dBm
rtl_power -f 98.1M:98.5M:50k -i 10 -g 40 -1 output.csv 
