#!/bin/bash

# Check if a directory path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

directory_path="$1"

# Count the number of subdirectories
num_subdirectories=$(find "$directory_path" -maxdepth 1 -mindepth 1 -type d | wc -l)

# Add 1 to the count
result=$((num_subdirectories + 1))

echo "Number of subdirectories in $directory_path: $result"
