#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 20:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=20G

# Define dirs
export MY_DIR=$HOME/jim_injections/tidal
export base_dir=$MY_DIR/new_slurm

# # Prepare getting the correct directory number
# echo "Getting the correct directory number from directory: $base_dir"
# latest_dir=$(find "$base_dir" -type d -name "injection_*" | sort -V | tail -n 1)
# latest_number=$(echo "$latest_dir" | grep -oE '[0-9]+')
# new_number=$((latest_number + 1))
# echo "New number: $new_number"
# new_dir_name="injection_$new_number"

# Count the number of subdirectories
num_subdirectories=$(find "$base_dir" -maxdepth 1 -mindepth 1 -type d | wc -l)
new_number=$((num_subdirectories + 1))

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
    --n-local-steps 100 \
    --n-loop-training 2 \
    --n-loop-production 2 \
    --eps-mass-matrix 0.0001 \
    --stopping-criterion-global-acc 0.10 \
    --waveform-approximant TaylorF2 \
 
export final_output_dir="$MY_DIR/new_slurm/$new_dir_name"
echo "Copying to: $final_output_dir"
#Copy output directory from scratch to home
cp -r "$TMPDIR/injection_2/" $final_output_dir

# Also copy the output file there
echo "Finally, moving the output file"
mv "$MY_DIR/slurm-$SLURM_JOBID.out" "$final_output_dir/log.out"

echo "DONE"