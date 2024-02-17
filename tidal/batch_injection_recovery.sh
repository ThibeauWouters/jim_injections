#!/bin/bash

# Set the number of times to run the script
num_times=30

# Loop to run the script "injection_recovery.py" the specified number of times
for ((i=1; i<=$num_times; i++)); do
    echo "===== Iteration $i ====="
    python injection_recovery.py
done
