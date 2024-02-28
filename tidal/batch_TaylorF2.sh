#!/bin/bash

# Set the number of times to run the script
num_times=100

# Loop to run the injection script
for ((i=1; i<=$num_times; i++)); do
    echo "===== Iteration $i ====="
    python old_injection_recovery.py \
        --outdir ./outdir_TaylorF2_part3/ \
        --relative-binning-binsize 100 \
        --stopping-criterion-global-acc 0.20 \
        --waveform-approximant TaylorF2
done
