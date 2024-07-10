"""
This script is designed to generate clear dataset from TEDLIUM dataset.
The new dataset will be saved as pickle file in the given output path.
The pickle file consists of a list of tuples. the first element in the tuple corresponds to the clapped spectogram of 
the minimal pair word-part from the audio file. the second element is either 1 or 0. 1 for verb and 0 for noun/adjective.
This labling is due the assumption that first stress (FS) corresponds to noun/adjective and inital stress (IS) corresponds to verb.
"""

import os
import spacy
import csv
import librosa
import pickle
from words import words_to_check

def get_timestamps(csv_path: str, line_in_csv: int):
    start_time = -1
    end_time = -1
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        start_time = csv_reader[line_in_csv][1]
        end_time = csv_reader[line_in_csv][2]
    return int(start_time), int(end_time)

def get_clapped_spectorgram(sph_path: str, start_time: int ,end_time: int):
    y, sr = librosa.load(sph_path, sr=None)

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    y_segment = y[start_sample:end_sample]
    spectogram = librosa.stft(y_segment)
    return spectogram
        
def update_samples(samples: list, sph_dir_path: str, file_name: str, nlp):
    csv_path = f"{sph_dir_path}/{file_name}w2v2_align.csv"
    trans_path = f"{sph_dir_path}/{file_name}.lab"
    sph_path = f"{sph_dir_path}/{file_name}.sph"

    with open(trans_path, "r") as trans_file:
        for line in trans_file:
            doc = nlp(line)
            for line, token in enumerate (doc):
                if token.text in words_to_check:
                    if token.pos_ in ('VERB'):
                        start_time, end_time = get_timestamps(csv_path, line)
                        if (start_time is not -1) and (end_time is not -1):
                            curr_spectogram = get_clapped_spectorgram(sph_path, start_time, end_time)
                            samples.append((curr_spectogram, 1))

                    elif token.pos_ in ('NOUN', 'PROPN', 'ADJ'):
                        curr_spectogram = get_clapped_spectorgram(sph_path, start_time, end_time)
                        samples.append((curr_spectogram, 0))
                    else:
                        pass

def generate_dataset(sph_dir_path: str, stm_dir_path: str):
    samples = []

    spacy.require_gpu()
    nlp = spacy.load("en_core_web_sm")

    for stm_file in os.listdir(stm_dir_path):
        if stm_file.endswith(".stm"):
            file_name = stm_file.split(".")[0]
            update_samples(samples, sph_dir_path, file_name, nlp)

    return samples
            
if __name__ == '__main__':
    # Receive input from user
    sph_dir_path = input("Enter path for sph dir: ")
    stm_dir_path = input("Enter path for stm dir: ")
    output_path = input("Enter path for pikle file location: ")

    dataset = generate_dataset(sph_dir_path, stm_dir_path)

    # extract tagged_recordings into pickle file                             
    pickle_file_path = f"{output_path}/TEDLIUM_clear.pkl"
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(dataset, file)
    