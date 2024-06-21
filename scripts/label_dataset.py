"""
This script is designed to label the data.
The script receives a sentence path to dir contains .FLAC files and text file that contains the transcript for each audio file (.FLAC).
For each recording, the features will be the embeddings from either whisper or wav2vec2 (can be configured), and the label will be:
1 - if the stress is at the start (e.g. PERfect)
0 - if the stress is at the end (e.g. perFECT)
-1 - if the audio does not include any word from (words_to_cehck) or if the minimal pair words is not either noun,verb or adjective.

make sure you use pip install! pip install git+https://github.com/openai/whisper.git for installing whisper.
This version will be used in order to get timestamps for getting embeddings corresponding to the minimal pair word.
make sure you use pip install jiwer as well.

Notice: if you want to use whisper model for embeddings extraction, you need to use pip install git+https://github.com/isaacOnline/whisper.git@extract-embeddings
for installing whisper.
This whipder branch includes an embaddings extraction feature.

"""

import whisper
import os
import pickle
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
from generate_dataset import words_to_check
# import matplotlib.pyplot as plt
import spacy

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

def get_start_end_times(audio_path: str)-> tuple:
    model = whisper.load_model("base")
    transcript = model.transcribe(word_timestamps=True, audio=file)
    for segment in transcript["segments"]:
        for word in segment["words"]:
          clean_word = remove_non_alphabetic(word["word"])
          if clean_word in words_to_check:
            new_start = float(word['start'])
            new_end = float(word['end'])
            print(f"|word:{clean_word}|start:{new_start}|end:{new_end}|")
            return new_start, new_end
            break
    return -1, -1

def get_embedding_from_whisper(audio_path: str)->tuple:
    model = whisper.load_model("base")

    # Transcribe audio to get transcript with word timestamps
    transcript = model.transcribe(audio=audio_path)
    return transcript["segments"][0]['encoder_embeddings']

def get_embedding_from_wav2vec2(audio_path: str)-> tuple:
    # Load pre-trained processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    # Load the audio file
    audio_file = audio_path
    audio, sample_rate = librosa.load(audio_file, sr=16000)

    start_time, end_time = get_start_end_times('/content/19-227-0013.flac')
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
    # def plot_embeddings(embeddings):
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     img = ax.imshow(embeddings.squeeze().T, aspect='auto', origin='lower')
    #     ax.set_title("Embeddings from [2, 3] seconds of audio")
    #     ax.set_xlabel("Time steps")
    #     ax.set_ylabel("Feature dimensions")
    #     fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
    #     fig.tight_layout()
    #     plt.show()

    # # Plot the embeddings
    # plot_embeddings(embeddings_for_time_window)
    return embeddings_for_time_window

def get_label(sentence: str)->int:
    nlp = spacy.load("en_core_web_sm") 
    doc = nlp(sentence)
    for token in doc:
        if token.text in words_to_check:
            if token.pos_ in ('VERB'):
                return 1
            elif token.pos_ in ('NOUN', 'ADJ'):
                return 0
            else:
                return -1
    return -1

def lebel_data(dataset_path: str, use_whisper_embedding = False):
  tagged_recordings = []
  for file_name in os.listdir(dataset_path):
    if file_name.endswith(".flac"):

        # Get embedding for current flac file
        if(use_whisper_embedding):
            curr_embedding = get_embedding_from_whisper(f"{dataset_path}/{file_name}")
        else:
            curr_embedding = get_embedding_from_whisper(f"{dataset_path}/{file_name}")

        # Collect relevant trans.txt content
        text_file_path = f"{dataset_path}/trans.txt"
        if os.path.exists(text_file_path):
            with open(text_file_path, "r") as trans_file:
                # Add relevant lines to new_trans_file
                for line in trans_file:
                    if file_name[:-5] in line:
                        parts = line.split(' ', 1)
                        if len(parts) > 1:
                            curr_label = get_label(parts[1])
                            if curr_label == 1 or curr_label == 0:
                                tagged_recordings.append((curr_embedding ,curr_label))

  # extract tagged_recordings into pickle file                             
  pickle_file_path = f"/{dataset_path}/tagged_data.pkl"
  with open(pickle_file_path, 'wb') as file:
    pickle.dump(tagged_recordings, file)        

if __name__ == '__main__':
    # Receive input from user
    dataset_path = input("Enter path for dataset: ")
    lebel_data(dataset_path)

