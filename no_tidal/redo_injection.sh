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
    --load-existing-config True \
    --stopping-criterion-global-acc 0.20 \
    --which-distance-prior powerlaw \
 
export final_output_dir="$MY_DIR/redo_slurm/injection_$injection_number$SLURM_JOB_NAME"
echo "Copying to: $final_output_dir"

#Copy output directory from scratch to home, but first check if exists
if [ -d "$final_output_dir" ]; then
    echo "Directory already exists: $final_output_dir"
else
    mkdir "$final_output_dir"
    echo "Directory created: $final_output_dir"
fi
cp -r $TMPDIR/injection_$injection_number/* $final_output_dir

# Also copy the output file there
echo "Finally, moving the output file"s
mv "$MY_DIR/slurm-$SLURM_JOBID.out" "$final_output_dir/log.out"

echo "DONE"