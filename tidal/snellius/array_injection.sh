#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 01:15:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=10G
#SBATCH --array=1-3

now=$(date)
echo "$now"
echo "Running new injection"

# Define dirs
export MY_DIR=$HOME/jim_injections/tidal
export base_dir=$MY_DIR/outdir_NRTv2_snellius
export script_name="old_injection_recovery_v2_snellius.py"

echo "New number: $SLURM_ARRAY_TASK_ID"

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
conda activate jim
 
# Copy necessary files
cp $MY_DIR/$script_name "$TMPDIR"
cp $MY_DIR/utils.py "$TMPDIR"
cp -r $MY_DIR/psds/ "$TMPDIR"

# Run the script
python $MY_DIR/$script_name \
    --outdir $TMPDIR \
    --stopping-criterion-global-acc 0.20 \
    --waveform-approximant IMRPhenomD_NRTidalv2 \

export final_output_dir="$MY_DIR/outdir_NRTv2_snellius/injection_$SLURM_ARRAY_TASK_ID"
echo "Copying to: $final_output_dir"

#Copy output directory from scratch to home, but first check if exists
if [ -d "$final_output_dir" ]; then
    echo "Directory already exists: $final_output_dir"
else
    mkdir "$final_output_dir"
    echo "Directory created: $final_output_dir"
fi
cp -r $TMPDIR/injection_2/* $final_output_dir

# # Also copy the output file there
echo "Finally, moving the output file"
mv "$MY_DIR/slurm-$SLURM_ARRAY_TASK_ID.out" "$final_output_dir/log.out"

echo "DONE"