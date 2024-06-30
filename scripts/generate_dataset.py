"""
This script is designed to generate clear dataset from LibriSpeech dataset.
Each new directory in the new dataset contains the fllowing:
    1.FLAC file: a recording that contains excatly one minimal pair word for words_to_check (see below).
    2. text file corresponds to the audio flile (transcript)
The script receives a path to a directory downloaed from LibriSpeech, bool flag and an output_dir_path and does the following:
1. Removes any FLAC files that does not contain excatly 1 minimal pair from words_to_check.
2. Removes any sub-subfolder that does not contain FLAC files.
3. Removes any subfolder that becomes empty after its sub-subfolders are removed.
4. Copies all remaining FLAC files from the sub-subfolders to  output_dir_path.
5. Collects and consolidates the relevant lines from each trans.txt file into a single trans.txt file in the output_dir_path.

"""
import os
import shutil
# import soundfile as sf

# Minimal pairs to check
words_to_check = [
    'present', 'presents', 'PRESENT', 'PRESENTS', 
    'record', 'records', 'RECORD', 'RECORDS', 
    'conduct', 'conducts', 'CONDUCT', 'CONDUCTS', 
    'permit', 'permits', 'PERMIT', 'PERMITS', 
    'conflict', 'conflicts', 'CONFLICT', 'CONFLICTS', 
    'content', 'contents', 'CONTENT', 'CONTENTS', 
    'increase', 'increases', 'INCREASE', 'INCREASES', 
    'decrease', 'decreases', 'DECREASE', 'DECREASES', 
    'object', 'objects', 'OBJECT', 'OBJECTS', 
    'convert', 'converts', 'CONVERT', 'CONVERTS', 
    'export', 'exports', 'EXPORT', 'EXPORTS', 
    'import', 'imports', 'IMPORT', 'IMPORTS', 
    'protest', 'protests', 'PROTEST', 'PROTESTS', 
    'suspect', 'suspects', 'SUSPECT', 'SUSPECTS', 
    'digest', 'digests', 'DIGEST', 'DIGESTS', 
    'reject', 'rejects', 'REJECT', 'REJECTS', 
    'perfect', 'perfects', 'PERFECT', 'PERFECTS', 
    'insult', 'insults', 'INSULT', 'INSULTS', 
    'progress', 'progresses', 'PROGRESS', 'PROGRESSES', 
    'refuse', 'refuses', 'REFUSE', 'REFUSES', 
    'extract', 'extracts', 'EXTRACT', 'EXTRACTS', 
    'rebel', 'rebels', 'REBEL', 'REBELS', 
    'address', 'addresses', 'ADDRESS', 'ADDRESSES', 
    'subject', 'subjects', 'SUBJECT', 'SUBJECTS', 
    'project', 'projects', 'PROJECT', 'PROJECTS', 
    'contrast', 'contrasts', 'CONTRAST', 'CONTRASTS', 
    'transfer', 'transfers', 'TRANSFER', 'TRANSFERS', 
    'entrance', 'entrances', 'ENTRANCE', 'ENTRANCES'
]

# Function to check if a directory contains any FLAC files
def contains_flac_files(directory: str)->bool:
    for file_name in os.listdir(directory):
        if file_name.endswith(".flac"):
            return True
    return False

# Moves all relevnt FLAC file to the new directory
# FLAC files will be clliped if one_word_clipped is True
# Creates new text file in and new directory
# Removing flac files with zero or more than 1 minimal pairs by lexical stress
def rearrange(current_dir: str, output_dir_path:str, new_trans_file):
    # Iterate over each subfolder in the current directory
    for subfolder_name in os.listdir(current_dir):
        subfolder_path = os.path.join(current_dir, subfolder_name)
        
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
                            if count == 1:

                                # Move FLAC file to clean_path
                                shutil.copy(flac_filepath, output_dir_path)
                                # Collect relevant trans.txt content
                                trans_txt_path = os.path.join(sub_subfolder_path, f"{subfolder_name}-{sub_subfolder_name}.trans.txt")
                                if os.path.exists(trans_txt_path):
                                    with open(trans_txt_path, "r") as trans_file:
                                        # Add relevant lines to new_trans_file
                                        for line in trans_file:
                                            if flac_filename[:-5] in line:
                                                new_trans_file.writelines(line)

def generate_dataset(current_dir: str, output_dir_path: str):
    #os.makedirs(output_dir_path, exist_ok=True)
    with open(os.path.join(output_dir_path, "trans.txt"), "w") as new_trans_file:
        rearrange(current_dir=current_dir, output_dir_path=output_dir_path,new_trans_file=new_trans_file)

        # Read the lines from the text file and create a dictionary
        lines_dict = {}
        for line in new_trans_file:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                lines_dict[parts[0]] = parts[1]

    # Iterate over each FLAC file in the directory
    for file_name in os.listdir(output_dir_path):
        if file_name.endswith('.flac'):
            base_name = os.path.splitext(file_name)[0]
            if base_name in lines_dict:
                # Create a new directory for the FLAC file
                new_dir = os.path.join(output_dir_path, base_name)
                os.makedirs(new_dir, exist_ok=True)
                
                # Move the FLAC file into the new directory
                original_flac_path = os.path.join(output_dir_path, file_name)
                new_flac_path = os.path.join(new_dir, file_name)
                shutil.move(original_flac_path, new_flac_path)
                
                # Create a new text file with the corresponding line
                new_text_file_path = os.path.join(new_dir, f'{base_name}.txt')
                with open(new_text_file_path, 'w') as new_text_file:
                    new_text_file.write(lines_dict[base_name])

    print("Operation completed successfully.")        

if __name__ == '__main__':
    # Receive input from user
    current_dir = input("Enter path for current dataset: ")
    output_dir_path = input("Enter path for new directory: ")

    generate_dataset(current_dir, output_dir_path)