#!/bin/bash

# Set the number of times to run the script
num_times=100

# Loop to run the injection script
for ((i=1; i<=$num_times; i++)); do
    echo "===== Iteration $i ====="
    python old_injection_recovery_v2.py \
        --outdir ./outdir_NRTv2/ \
        --relative-binning-binsize 250 \
        --stopping-criterion-global-acc 0.20 \
        --waveform-approximant IMRPhenomD_NRTidalv2
done
