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
# Injection number to be ran
declare -i injection_number=14
export injection_number

echo "Running injection number: $injection_number"

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
conda activate jim
 
# Copy necessary files
cp $MY_DIR/injection_recovery.py "$TMPDIR"
cp $MY_DIR/utils.py "$TMPDIR"
cp -r $MY_DIR/psds/ "$TMPDIR"
# Now, also copy the original injection directory
cp -r "$MY_DIR/redo_slurm/injection_$injection_number" "$TMPDIR"

# Run the script
python $MY_DIR/injection_recovery.py \
    --outdir $TMPDIR \
    --N $injection_number \
    --n-local-steps 100 \
    --eps-mass-matrix 0.0001 \
    --stopping-criterion-global-acc 0.20 \
    --waveform-approximant TaylorF2 \
 
export final_output_dir="$MY_DIR/redo_slurm/injection_$injection_number"
echo "Copying to: $final_output_dir"
#Copy output directory from scratch to home
cp -r "$TMPDIR/injection_$injection_number/" $final_output_dir

# Also copy the output file there
echo "Finally, moving the output file"
mv "$MY_DIR/slurm-$SLURM_JOBID.out" "$final_output_dir/log.out"

echo "DONE"