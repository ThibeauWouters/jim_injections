#!/bin/bash

# Set the number of times to run the script
num_times=100

# Loop to run the injection script
for ((i=1; i<=$num_times; i++)); do
    echo "===== Iteration $i ====="
    python old_injection_recovery_v2_no_taper.py \
        --outdir ./outdir_NRTv2_ref_wf_no_taper/ \
        --stopping-criterion-global-acc 0.20 \
        --waveform-approximant IMRPhenomD_NRTidalv2
done