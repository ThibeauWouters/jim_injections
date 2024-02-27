python injection_recovery.py \
    --outdir ./redo_slurm/ \
    --N "14" \
    --load-existing-config True \
    --eps-mass-matrix 0.000005 \
    --stopping-criterion-global-acc 0.20 \
    --which-distance-prior powerlaw \
    --waveform-approximant TaylorF2 \