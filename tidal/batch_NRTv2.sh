#!/bin/bash

# Set the number of times to run the script
num_times=60

# Loop to run the injection script
for ((i=1; i<=$num_times; i++)); do
    echo "===== Iteration $i ====="
    python old_injection_recovery.py \
        --outdir ./outdir_NRTv2_binsize_1000/ \
        --relative-binning-binsize 1000 \
        --stopping-criterion-global-acc 0.20 \
        --which-local-sampler GaussianRandomWalk \
        --waveform-approximant IMRPhenomD_NRTidalv2
done
