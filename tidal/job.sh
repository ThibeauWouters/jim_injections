#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 20:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=20G

#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

conda activate jim

# input dirs
export MY_DIR=$HOME/jim_injections/tidal
export PSD_DIR=$HOME/jim_injections/tidal/psds

# output dir
export TMPDIR_OUTPUT=$TMPDIR/out_slurm/
 
# Copy input file and auxiliary files to scratch
cp $MY_DIR/injection_recovery.py "$TMPDIR"
cp $MY_DIR/utils.py "$TMPDIR"
cp -r $PSD_DIR "$TMPDIR"
 
#Create output directory on scratch
mkdir $TMPDIR_OUTPUT    
 
#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python $MY_DIR/injection_recovery.py \
    --outdir $TMPDIR_OUTPUT \
    --n-local-steps 100 \
    --eps-mass-matrix 0.01 \
    --stopping-criterion-global-acc 0.20 \
    --waveform-approximant TaylorF2 \
 
#Copy output directory from scratch to home
cp -r "$OUTPUT_DIR" "$MY_DIR"

echo "DONE"
pwd
ls -l $TMPDIR_OUTPUT

echo "DONE DONE"