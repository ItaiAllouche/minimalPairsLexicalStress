import os
import shutil

"""
This script is designed to clean up and consolidate audio and transcription files from a nested directory structure (e.g. LibriSpeech dataset) into a single
directory (dev_set_clean).
The script performs the following operations:
1. Removes any FLAC files that does not contain excatly 1 minimal pair from words_to_check (defined below)
2. Removes any sub-subfolder that does not contain FLAC files.
3. Removes any subfolder that becomes empty after its sub-subfolders are removed.
3. Copies all remaining FLAC files from the sub-subfolders to the dev_set_clean directory.
4. Collects and consolidates the relevant lines from each trans.txt file into a single trans.txt file in the dev_set_clean directory.
"""

# Path to the main directory containing subfolders
main_directory_path = "/mnt/c/Users/Owner/OneDrive/Desktop/Technion/semester8/project_b/minimalParisLexicalStress/dataset/dev_set"

# Minimal pairs to check
words_to_check = [
    'present', 'PRESENT', 'record', 'RECORD', 'conduct', 'CONDUCT', 'permit', 'PERMIT', 'conflict', 'CONFLICT',
    'content', 'CONTENT', 'increase', 'INCREASE', 'decrease', 'DECREASE', 'object', 'OBJECT', 'convert', 'CONVERT',
    'export', 'EXPORT', 'import', 'IMPORT', 'protest', 'PROTEST', 'suspect', 'SUSPECT', 'digest', 'DIGEST',
    'reject', 'REJECT', 'perfect', 'PERFECT', 'insult', 'INSULT', 'progress', 'PROGRESS', 'refuse', 'REFUSE',
    'extract', 'EXTRACT', 'rebel', 'REBEL', 'address', 'ADDRESS', 'subject', 'SUBJECT', 'project', 'PROJECT',
    'contrast', 'CONTRAST', 'transfer', 'TRANSFER', 'entrance', 'ENTRANCE'
]

# Function to check if a directory contains any FLAC files
def contains_flac_files(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith(".flac"):
            return True
    return False

# Removing flac files with zero or more than 1 minimal pairs by lexical stress
def rmv_unrelevant_flac_files():
    # Iterate over each subfolder in the main directory
    for subfolder_name in os.listdir(main_directory_path):
        subfolder_path = os.path.join(main_directory_path, subfolder_name)
        
        # Check if the current path is a directory
        if os.path.isdir(subfolder_path):
            # Iterate over each sub-subfolder in the current subfolder
            for sub_subfolder_name in os.listdir(subfolder_path):
                sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder_name)
                
                # Check if the current path is a directory
                if os.path.isdir(sub_subfolder_path):
                    # Path to the transcript file in the current sub-subfolder
                    transcript_path = os.path.join(sub_subfolder_path, f"{subfolder_name}-{sub_subfolder_name}.trans.txt")
                    
                    # Check if the transcript file exists
                    if os.path.exists(transcript_path):
                        # Read the transcript file
                        with open(transcript_path, "r") as transcript_file:
                            lines = transcript_file.readlines()
                        
                        # Iterate over the lines and corresponding FLAC files
                        for i, line in enumerate(lines):

                            # Remove leading/trailing whitespace and split the line into words
                            words_in_line = line.strip().split()

                            # Construct the file name
                            flac_filename = f"{words_in_line[0]}.flac"
                            flac_filepath = os.path.join(sub_subfolder_path, flac_filename)

                            # Count how many words from the list are in the current line
                            count = sum(1 for word in words_in_line if word in words_to_check)

                            # Check if exactly one word from the list is present in the line
                            if count != 1:

                                # Delete the FLAC file if it exists
                                if os.path.exists(flac_filepath):
                                    os.remove(flac_filepath)                                                                

# Removing folder without .flac files
def rmv_empty_folders():
    # Iterate over each subfolder in the main directory
    for subfolder_name in os.listdir(main_directory_path):
        subfolder_path = os.path.join(main_directory_path, subfolder_name)
        
        # Check if the current path is a directory
        if os.path.isdir(subfolder_path):
            # Flag to check if the subfolder is empty after removing sub-subfolders
            subfolder_empty = True
            
            # Iterate over each sub_subfolder in the current subfolder
            for sub_subfolder_name in os.listdir(subfolder_path):
                sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder_name)
                
                # Check if the current path is a directory
                if os.path.isdir(sub_subfolder_path):
                    # If the sub_subfolder does not contain any FLAC files, delete it
                    if not contains_flac_files(sub_subfolder_path):
                        # print(f"Removing empty sub-subfolder: {sub_subfolder_path}")
                        shutil.rmtree(sub_subfolder_path)
                    else:
                        # If the sub_subfolder contains FLAC files, mark the subfolder as not empty
                        subfolder_empty = False
            
            # If the subfolder is empty after removing sub-subfolders, delete it
            if subfolder_empty:
                shutil.rmtree(subfolder_path)

# Function to copy FLAC files to dev_set_clean directory and correspond transcript to new_trans_file
def copy_flac_and_trans(subfolder_name, subfolder_path, clean_path, new_trans_file):
    for sub_subfolder_name in os.listdir(subfolder_path):
        sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder_name)
        if os.path.isdir(sub_subfolder_path):
            for file_name in os.listdir(sub_subfolder_path):
                if file_name.endswith(".flac"):
                    # Move FLAC file to dev_set_clean
                    shutil.copy(os.path.join(sub_subfolder_path, file_name), clean_path)
                    # Collect relevant trans.txt content
                    trans_txt_path = os.path.join(sub_subfolder_path, f"{subfolder_name}-{sub_subfolder_name}.trans.txt")
                    if os.path.exists(trans_txt_path):
                        with open(trans_txt_path, "r") as trans_file:
                            # Add relevant lines to new_trans_file
                            for line in trans_file:
                                if file_name[:-5] in line:
                                    new_trans_file.writelines(line)

def rearrange():
    dev_set_clean_path = "/mnt/c/Users/Owner/OneDrive/Desktop/Technion/semester8/project_b/minimalParisLexicalStress/dataset/dev_set_clean"
    os.makedirs(dev_set_clean_path, exist_ok=True)

    # Merge trans.txt content and create dev_set_clean trans.txt
    with open(os.path.join(dev_set_clean_path, "trans.txt"), "w") as new_trans_file:
        for subfolder_name in os.listdir(main_directory_path):
            subfolder_path = os.path.join(main_directory_path, subfolder_name)
            if os.path.isdir(subfolder_path):
                for sub_subfolder_name in os.listdir(subfolder_path):
                    sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder_name)
                    if os.path.isdir(sub_subfolder_path):
                        copy_flac_and_trans(subfolder_name, subfolder_path, dev_set_clean_path, new_trans_file)

if __name__ == '__main__':
    rmv_unrelevant_flac_files()
    rmv_empty_folders()
    rearrange()