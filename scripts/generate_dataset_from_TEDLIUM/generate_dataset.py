"""
This script is designed to generate clear dataset from TEDLIUM dataset.
The new dataset will be saved as pickle file in the given output path.
The pickle file consists of a list of tuples. the first element in the tuple corresponds to the clapped spectogram of 
the minimal pair word-part from the audio file. the second element is either 1 or 0. 1 for verb and 0 for noun/adjective.
This labling is due the assumption that first stress (FS) corresponds to noun/adjective and inital stress (IS) corresponds to verb.
"""

import os
import spacy
import librosa
import pandas as pd
import re
# from words import words_to_check
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_suffix_regex, compile_infix_regex


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

def customize_tokenizer(nlp):
    # Customize tokenizer patterns
    prefixes = nlp.Defaults.prefixes
    suffixes = nlp.Defaults.suffixes

    # Modify the infix patterns to exclude apostrophes
    infixes = (
        [r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''']  # Default patterns, excluding the pattern for apostrophes
    )

    # Compile the regex for the custom infix patterns
    infix_re = compile_infix_regex(infixes)

    # Create a new tokenizer with the customized infix patterns
    nlp.tokenizer = Tokenizer(nlp.vocab, 
                            prefix_search=compile_prefix_regex(prefixes).search,
                            suffix_search=compile_suffix_regex(suffixes).search,
                            infix_finditer=infix_re.finditer,
                            token_match=None)
    return nlp

def get_timestamps(csv_path: str, line_in_csv: int):
    start_time = -1
    end_time = -1

    df = pd.read_csv(csv_path, header=None)
    row = df.iloc[line_in_csv]
    start_time = float(row[1])
    end_time = start_time + 0.5 # fixed-length segments, 0.5 sec each
    return float(start_time), float(end_time)

def get_clapped_spectorgram(sph_path: str, start_time: float ,end_time: float):
    y, sr = librosa.load(sph_path, sr=None)

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    y_segment = y[start_sample:end_sample]
    # spectogram = librosa.stft(y_segment, n_fft=4096, win_length=2048)
    return y_segment
        
def update_samples(samples: list, sph_dir_path: str, file_name: str, nlp, stress:dict):
    csv_path = f"{sph_dir_path}/{file_name}w2v2_align.csv" # csv file contains timestamps
    trans_path = f"{sph_dir_path}/{file_name}.lab" # transcript file
    sph_path = f"{sph_dir_path}/{file_name}.sph" # audio file

    with open(trans_path, "r") as trans_file:
        for line in trans_file:
            line = re.sub(r"(\w+)'(\w+)", r'\1\2', line)
            doc = nlp(line)
            for csv_line, token in enumerate (doc):
                if (token.text in words_to_check) and (csv_line < len(doc)-1):
                    if token.pos_ in ('VERB'):
                        start_time, end_time = get_timestamps(csv_path, csv_line)
                        if (start_time != -1) and (end_time != -1):
                            curr_spectogram = get_clapped_spectorgram(sph_path, start_time, end_time)
                            samples.append((curr_spectogram, 1))
                            stress['FS'] += 1

                    elif token.pos_ in ('NOUN', 'PROPN', 'ADJ'):
                        start_time, end_time = get_timestamps(csv_path, csv_line)
                        if (start_time != -1) and (end_time != -1):
                            curr_spectogram = get_clapped_spectorgram(sph_path, start_time, end_time)
                            samples.append((curr_spectogram, 0))
                            stress['IS'] += 1
                    else:
                        pass

def generate_dataset(sph_dir_path: str, stm_dir_path: str):
    samples = []
    stress = {"IS": 0, "FS": 0}

    spacy.require_gpu()
    nlp = spacy.load("en_core_web_sm")

    # prevent spliting by some tokens. E.g. did'nt -> did 'nt
    nlp = customize_tokenizer(nlp)

    for i, stm_file in enumerate(os.listdir(stm_dir_path)):
        print(f"file: {stm_file} | precentege: {100*(i+1)/2351}%")
        if stm_file.endswith(".stm"):
            file_name = stm_file.split(".stm")[0]
            update_samples(samples, sph_dir_path, file_name, nlp, stress)
            print(f"IS: {stress['IS']}, FS: {stress['FS']}")

    df = pd.DataFrame(samples, columns=['audio', 'label'])
    df.to_parquet('/app/new_datasets/TEDLIUM/labeld_audio.parquet', engine='pyarrow')
    print(f"IS: {stress['IS']}, FS: {stress['FS']}")
         
if __name__ == '__main__':
    # Receive input from user
    sph_dir_path = "/app/new_datasets/TEDLIUM/TEDLIUM_release-3/legacy/train/sph" #input("Enter path for sph dir: ")
    stm_dir_path = "/app/new_datasets/TEDLIUM/TEDLIUM_release-3/legacy/train/stm" #input("Enter path for stm dir: ")
    output_path = "/app/new_datasets/TEDLIUM"  #input("Enter path for pikle file location: ")

    generate_dataset(sph_dir_path, stm_dir_path)