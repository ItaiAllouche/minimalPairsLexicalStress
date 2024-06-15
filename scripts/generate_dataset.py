"""
This script is desigmed to generate new dataset from LibriSpeech dataset.
Each element in the new dataset contains the fllowing:
    1.FLAC file a recording that contains excatly
      one minimal pair word for words_to_check (see blow).
    2. label: 1 if the minimal pair word is verb, 0 in noun.
The script receives a path to a directory downloaed from LibriSpeech, bool flag and an output_dir_path and does the following:
1. Removes any FLAC files that does not contain excatly 1 minimal pair from words_to_check.
2. Removes any sub-subfolder that does not contain FLAC files.
3. Removes any subfolder that becomes empty after its sub-subfolders are removed.

if the flag set to false:
4. Copies all remaining FLAC files from the sub-subfolders to  output_dir_path.

if the flag set to true:
4. From each remaining FLAC file, clliped it to contain only the minimal pair word.

5. Collects and consolidates the relevant lines from each trans.txt file into a single trans.txt file in the output_dir_path.

make sure you use pip install! pip install git+https://github.com/openai/whisper.git for installing whisper.
This version will be used in order to get timestamps for clapping the FLAC file.
make sure you use pip install jiwer as well.
"""

import os
import shutil
import soundfile as sf
import whisper

# Minimal pairs to check
words_to_check = [
    'present', 'PRESENT', 'record', 'RECORD', 'conduct', 'CONDUCT', 'permit', 'PERMIT', 'conflict', 'CONFLICT',
    'content', 'CONTENT', 'increase', 'INCREASE', 'decrease', 'DECREASE', 'object', 'OBJECT', 'convert', 'CONVERT',
    'export', 'EXPORT', 'import', 'IMPORT', 'protest', 'PROTEST', 'suspect', 'SUSPECT', 'digest', 'DIGEST',
    'reject', 'REJECT', 'perfect', 'PERFECT', 'insult', 'INSULT', 'progress', 'PROGRESS', 'refuse', 'REFUSE',
    'extract', 'EXTRACT', 'rebel', 'REBEL', 'address', 'ADDRESS', 'subject', 'SUBJECT', 'project', 'PROJECT',
    'contrast', 'CONTRAST', 'transfer', 'TRANSFER', 'entrance', 'ENTRANCE'
]

def clip_audio(input_file, output_file, new_start, new_end):
    # Read the input FLAC audio file
    audio, sample_rate = sf.read(input_file)
    strach = 0.28 # in [sec]

    # Convert start and end times from seconds to samples
    # strach the end if possible
    new_end = new_end + strach if ((new_end+strach)*sample_rate <= len(audio))\
                               else -1
    start_sample = int(new_start * sample_rate)
    end_sample = int(new_end * sample_rate)

    # Clip the audio between start and end samples
    if new_end == -1:
      clipped_audio = audio[start_sample:]
    else:
      clipped_audio = audio[start_sample:end_sample]

    # Save the clipped audio to a new FLAC file
    sf.write(output_file, clipped_audio, sample_rate)

def remove_non_alphabetic(input_string):
    # Initialize an empty string to store the result
    result = ""

    # Iterate over each character in the input string
    for char in input_string:
        # Check if the character is an alphabet letter
        if char.isalpha():
            # If it is, append it to the result string
            result += char
    return result

# Function to check if a directory contains any FLAC files
def contains_flac_files(directory: str)->bool:
    for file_name in os.listdir(directory):
        if file_name.endswith(".flac"):
            return True
    return False

def clip_audio_to_one_word(output_dir_path):
    model = whisper.load_model("base")
    for file_name in os.listdir(output_dir_path):
        if file_name.endswith(".flac"):
            transcript = model.transcribe(word_timestamps=True, audio=f"{output_dir_path}/{file_name}")
            for segment in transcript["segments"]:
                for word in segment["words"]:
                    clean_word = remove_non_alphabetic(word["word"])
                    new_start = float(word['start'])
                    new_end = float(word['end'])
                    if clean_word in words_to_check:
                        print(f"|word:{clean_word}|start:{new_start}|end:{new_end}|")
                        clip_audio(input_file=f"{output_dir_path}/{file_name}",
                                    output_file=f"{output_dir_path}/{file_name}",
                                    new_start=new_start, new_end=new_end)
                        break

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

def generate_dataset(current_dir: str, output_dir_path: str, one_word_clipped=False):
    os.makedirs(output_dir_path, exist_ok=True)
    with open(os.path.join(output_dir_path, "trans.txt"), "w") as new_trans_file:
        rearrange(current_dir=current_dir, output_dir_path=output_dir_path,new_trans_file=new_trans_file)
        if one_word_clipped:
            clip_audio_to_one_word(output_dir_path=output_dir_path)

if __name__ == '__main__':
    # Receive input from user
    current_dir = input("Enter path for current dataset: ")
    output_dir_path = input("Enter path for new directory: ")
    one_word_clipped = bool(input("Enter 1 for clip audio file to contain only 1 word, else 0: "))

    generate_dataset(current_dir, output_dir_path, one_word_clipped)