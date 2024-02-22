#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 40:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=25G

now=$(date)
echo "$now"
echo "Running new injection"

# Define dirs
export MY_DIR=$HOME/jim_injections/tidal
export base_dir=$MY_DIR/new_slurm

# Count the number of subdirectories
num_subdirectories=$(find "$base_dir" -maxdepth 1 -mindepth 1 -type d | wc -l)
new_number=$((num_subdirectories + 1))

echo "New number: $new_number"

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
conda activate jim
 
# Copy necessary files
cp $MY_DIR/injection_recovery.py "$TMPDIR"
cp $MY_DIR/utils.py "$TMPDIR"
cp -r $MY_DIR/psds/ "$TMPDIR"

# Run the script
python $MY_DIR/injection_recovery.py \
    --outdir $TMPDIR \
    --n-local-steps 50 \
    --eps-mass-matrix 0.0001 \
    --stopping-criterion-global-acc 0.15 \
    --waveform-approximant TaylorF2 \
 
export final_output_dir="$MY_DIR/new_slurm/injection_$new_number"
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
mv "$MY_DIR/slurm-$SLURM_JOBID.out" "$final_output_dir/log.out"

echo "DONE"