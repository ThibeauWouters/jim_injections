python save_likelihoods.py \
    --outdir ./outdir/ \
    --load-existing-config True \
    --N 14 \
    --stopping-criterion-global-acc 0.20 \
    --waveform-approximant TaylorF2 \

# python compare_likelihoods.py --outdir ./outdir/ --load-existing-config True --N 14 --stopping-criterion-global-acc 0.20  --waveform-approximant TaylorF2