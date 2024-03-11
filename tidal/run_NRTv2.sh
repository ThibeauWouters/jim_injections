python old_injection_recovery_v2_no_taper.py \
    --outdir ./outdir_NRTv2_binsize_1000 \
    --N "4_no_taper_ref_wf" \
    --load-existing-config True \
    --save-likelihood True \
    --relative-binning-binsize 1000 \
    --stopping-criterion-global-acc 0.20 \
    --waveform-approximant IMRPhenomD_NRTidalv2