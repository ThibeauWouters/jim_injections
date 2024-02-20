#!/bin/bash

# Set the number of times to run the script
num_times=100

# Loop to run the script "injection_recovery.py" the specified number of times
for ((i=1; i<=$num_times; i++)); do
    echo "===== Iteration $i ====="
    python injection_recovery.py \
        --outdir ./outdir_TaylorF2_part2/ \
        --stopping-criterion-global-acc 0.20 \
        --waveform-approximant TaylorF2 \
        --relative-binning-binsize 500
done
