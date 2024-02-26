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
export injection_dir=$HOME/jim_injections
export events_dir=$HOME/jim_injections/real_events
export data_dir=$HOME/jim_injections/real_events/data/GW170817/
export tidal_dir=$HOME/jim_injections/tidal

export this_event_dir=$events_dir/GW170817_TaylorF2

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
conda activate jim
 
# Copy necessary files, in this case, the data to analyse
# the tempdir will look as follows at the end of the run:
# --- $TMPDIR
#     --- analyze_event.py
#     --- utils.py
#     --- data/
#         --- datafiles come here
#     --- outdir/
#         --- output files come here

cp $this_event_dir/analyze_event.py "$TMPDIR"
cp $tidal_dir/utils.py "$TMPDIR"
if [ -d "$TMPDIR/data/" ]; then
    echo "Directory already exists: $TMPDIR/data/"
else
    mkdir "$TMPDIR/data/"
    echo "Directory created: $TMPDIR/data/"
fi
cp -r $data_dir/* $TMPDIR/data/

# Run the script
python $TMPDIR/analyze_event.py
 
# Copy the output directory from scratch to home
export final_output_dir=$this_event_dir/outdir/
# echo "Copying to: $final_output_dir"
# # First check if exists
# if [ -d "$final_output_dir" ]; then
#     echo "Directory already exists: $final_output_dir"
# else
#     mkdir "$final_output_dir"
#     echo "Directory created: $final_output_dir"
# fi
# cp $TMPDIR/outdir/* $final_output_dir

# # Also copy the output file there
echo "Finally, moving the output file"
mv "$events_dir/slurm-$SLURM_JOBID.out" "$final_output_dir/log.out"

echo "DONE"