"""
This script is designed to label the data.
The script receives path to dir that contains dirs, each dir contains:
1. .FLAC files (audio) 
2. .txt file - contains the transcript the audio file (.FLAC).
3. .TextGrid file- contains relevant data for time-stamps extraction.
For each recording, the features will be the embeddings from either whisper or wav2vec2 (can be configured), and the label will be:
1 - if the stress is at the start (e.g. PERfect)
0 - if the stress is at the end (e.g. perFECT)
-1 - if the audio does not include any word from (words_to_cehck) or if the minimal pair words is not either noun,verb or adjective.

Notice: if you want to use whisper model for embeddings extraction, you need to use pip install git+https://github.com/isaacOnline/whisper.git@extract-embeddings
for installing whisper.
This whipder branch includes an embaddings extraction feature.

"""

# import whisper
import textgrid as tg
import os
import pickle
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
from generate_dataset import words_to_check
import matplotlib.pyplot as plt
import spacy
import subprocess
import torch.nn.functional as F

def remove_non_alphabetic(input_string: str)->str:
    # Initialize an empty string to store the result
    result = ""

    # Iterate over each character in the input string
    for char in input_string:
        # Check if the character is an alphabet letter
        if char.isalpha():
            # If it is, append it to the result string
            result += char
    return result

def read_textgrid_data(TextGrid_path: str)->list:

    textgrid_file = tg.TextGrid.fromFile(TextGrid_path)
    intervals = []

    for interval in textgrid_file[0]: #0 for words, 1 for phones

        xmin = interval.minTime
        xmax = interval.maxTime
        text = interval.mark

        intervals.append((xmin, xmax, text))

    return intervals

# TODO: check if stach is necessery
def get_time_stamps(TextGrid_path: str)-> tuple:
    sentence_time_stamps = read_textgrid_data(TextGrid_path)
    for segment in sentence_time_stamps:
        if segment[2] in words_to_check:
            return float(segment[0]), float(segment[1])
    return -1, -1

# reshape embeddings size to fixed size using bicubic interpolation. 
def fix_embedding_size(emb: tuple, fix_size: tuple)->tuple:
    if emb.dim() != 3 or emb.size(0) != 1:
        raise ValueError("Input data must have shape (1, height, width).")

    # Convert new shape to match the expected input for interpolate
    new_height, new_width = fix_size[1], fix_size[2]
    
    # Perform the interpolation using torch.nn.functional.interpolate
    reshaped_data = F.interpolate(emb.unsqueeze(0), size=(new_height, new_width), mode='bicubic', align_corners=False).squeeze(0)
    
    return reshaped_data

# Complete if neccesery, currently not supported.
def get_embedding_from_whisper(dir_path: str, dir_name: str)->tuple:
    model = whisper.load_model("base")

    # Transcribe audio to get transcript with word timestamps
    transcript = model.transcribe(audio=audio_path)
    return transcript["segments"][0]['encoder_embeddings']

def get_embedding_from_wav2vec2(dir_path: str, dir_name: str, processor, model, plot = False)-> tuple:
    # Load the audio file
    audio_file = f'{dir_path}/{dir_name}.flac'
    audio, sample_rate = librosa.load(audio_file, sr=16000)

    start_time, end_time = get_time_stamps(f'{dir_path}/{dir_name}.TextGrid')
    strach = 0.28 # in [sec]
    end_time = end_time + strach if ((end_time+strach)*sample_rate <= len(audio))\
                                 else end_time
    
    # Process the full audio with the Wav2Vec2 processor
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the last hidden state
    last_hidden_state = outputs.last_hidden_state

    # Calculate the corresponding sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Calculate the number of feature vectors per second
    input_length_in_sec = len(audio) / sample_rate
    num_vectors = last_hidden_state.shape[1]
    vectors_per_second = num_vectors / input_length_in_sec

    # Calculate the indices for the time window
    start_vector = int(start_time * vectors_per_second)
    end_vector = int(end_time * vectors_per_second)

    # Extract the embeddings for the time window
    embeddings_for_time_window = last_hidden_state[:, start_vector:end_vector, :]

    # Optionally, plot the embeddings for visualization
    def plot(em):
        pass
        # fig, ax = plt.subplots(1,1, figsize=(14, 8))

        # # Plot all embedding
        # img = ax.imshow(em.squeeze().T, aspect='auto', origin='lower', vmax=3, vmin=-3)
        # ax.set_title(f"Embeddings")
        # ax.set_ylabel("Feature dimensions")
        # fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")

        # fig.tight_layout()
        # plt.show()

    if(plot):
        # Plot the embeddings
        plot(embeddings_for_time_window)

    return embeddings_for_time_window

def get_label(txt_path: str, nlp)-> int:
    if os.path.exists(txt_path):
        with open(txt_path, "r") as trans_file:
            for line in trans_file:
                doc = nlp(line)
                for token in doc:
                    if token.text in words_to_check:
                        if token.pos_ in ('VERB'):
                            #word_count[token.text] = (word_count[token.text][0] + 1, word_count[token.text][1], word_count[token.text][2])
                            return 1
                        elif token.pos_ in ('NOUN', 'ADJ', 'PROPN'):
                            #word_count[token.text] = (word_count[token.text][0], word_count[token.text][1] + 1, word_count[token.text][2])
                            return 0
                        else:
                            return -1
    return -1

def lebel_data(dataset_path: str, use_whisper_embedding = False):
    # two dataset for testing comparation.
    # tagged_fixed_embeddings contains embeddings of equal shape.
    tagged_embeddings = []
    tagged_fixed_embeddings = []

    # embeddings fixed size, min value and max value
    fix_size = (1,57,768)
    vmin = -4
    vmax = 4

    # loading wav2vec2 model the its processor if necessery.
    if use_whisper_embedding is False:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    
    # download and load english model for spacy.
    # here we use a small model.
    spacy.require_gpu()
    # subprocess.run('python -m spacy download en_core_web_sm', shell=True, check=True)
    nlp = spacy.load("en_core_web_sm")

    for dir_name in os.listdir(dataset_path):
        if not dir_name.endswith(".txt"):
            # Get embedding for current dir
            if use_whisper_embedding:
                curr_embedding = get_embedding_from_whisper(f'{dataset_path}/{dir_name}')
            else:
                curr_embedding = get_embedding_from_wav2vec2(f'{dataset_path}/{dir_name}', dir_name, processor, model)
                # clip embedding values to be between [vmin,vmax]
                curr_embedding = curr_embedding.clamp(min=vmin, max=vmax)
            
            curr_label = get_label(f'{dataset_path}/{dir_name}/{dir_name}.txt', nlp)
            if curr_label in (0, 1):
                tagged_embeddings.append((curr_embedding ,curr_label))
                tagged_fixed_embeddings.append((fix_embedding_size(curr_embedding, fix_size).clamp(min=vmin, max=vmax) ,curr_label))

    # extract tagged_recordings into pickle file                             
    pickle_file_path = f"./tagged_embeddings.pkl"
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(tagged_embeddings, file)
        # extract tagged_recordings into pickle file   
                          
    pickle_file_path = f"./tagged_fixed_embeddings.pkl"
    with open(pickle_file_path, 'wb') as file:
        pickle.dump(tagged_fixed_embeddings, file)   

if __name__ == '__main__':
    # Receive input from user
    dataset_path = input("Enter path for dataset: ")
    lebel_data(dataset_path)
