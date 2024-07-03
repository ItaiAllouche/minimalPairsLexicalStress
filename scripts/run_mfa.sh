#!/bin/bash
# This script is desinged to use the montreal-forced-aligner package in order to extract time-stamps for the audio files.
# running this script, will add .TextGrid file to each dir that contains audio files and the corresponding .txt file (transcript).
# time stamps will get extracted using label_dataset.py script by using the TextGrid file.

#install and activate conda
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
# source ~/miniconda3/bin/activate -y 
# conda init
# source ~/.bashrc -y 
# conda create -n aligner -c conda-forge montreal-forced-aligner -y
# conda activate aligner
# mfa model download acoustic english_us_arpa
# mfa model download dictionary english_us_arpa 

cd ../datasets/train_100_clean || { echo "Error: ../datasets directory not found"; exit 1; }

GPU_DEVICE_ID=7

# Iterate through each directory in the current directory
for dir in */; do
    # Trim trailing slash to get directory name without path
    dir_name="${dir%/}"
    
    # Check if the item is a directory
    if [[ -d "$dir" ]]; then
        # Check if the directory is not the script's own directory
        if [[ "$dir_name" != "$(basename "$0")" ]]; then
            # Check if no .TextGrid file exists in the directory
            if ! ls "$dir"/*.TextGrid 1> /dev/null 2>&1; then
                # Run mfa align in each directory
                echo "Processing directory: $dir_name"
                mfa align --clean "$dir" english_us_arpa english_us_arpa "$dir" -gpu "$GPU_DEVICE_ID"
            else
                echo "Skipping directory $dir_name: .TextGrid file already exists"
            fi
        fi
    fi
done

conda deactivate
conda deactivate