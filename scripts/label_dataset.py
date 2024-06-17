"""
This script is designed to label the data.
The script receives a sentence (text).
Returns 1 if the minimal pair in the sentence is a verb, 0 if it is a noun, and -1 if the sentence does not contain a minimal pair from words_to_check
or the word did not tagged as verb or noun.

make sure you use pip install git+https://github.com/isaacOnline/whisper.git@extract-embeddings for installing whisper.
This branch has an embaddings extraction feature which will be use in the labling process.
"""

import nltk
import whisper
import os
import nltk
import pickle
import numpy as np
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from generate_dataset import words_to_check

def get_embedding(audio_path: str):
    model = whisper.load_model("base")

    # Transcribe audio to get transcript with word timestamps
    transcript = model.transcribe(audio=audio_path)
    return transcript["segments"][0]['encoder_embeddings']

def get_label(sentence: str)->int:
    text = nltk.word_tokenize(sentence)
    pos_tagged = nltk.pos_tag(text)
    for curr_set in pos_tagged:
        if curr_set[0] in words_to_check:
            if curr_set[1] in ('VB', 'VBG', 'VBP', 'VBZ','VBD', 'VBN'):
                return 1
            elif curr_set[1] in ('NN', 'NNP'):
                return 0
            else:
                return -1
    return -1

def lebel_data(dataset_path: str):
  tagged_recordings = []
  for file_name in os.listdir(dataset_path):
    if file_name.endswith(".flac"):

        # Get embedding for current flac file
        curr_embedding = get_embedding(f"{dataset_path}/{file_name}")

       # Select the final layer's embeddings
        curr_embedding = curr_embedding[:, -1, :, :]

        # Remove the batch dimension
        curr_embedding = curr_embedding.squeeze(0)

        # Compute the mean across the sequence dimension
        curr_embedding = np.mean(curr_embedding, axis=0)

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
                            tagged_recordings.append((curr_embedding ,curr_label))

  # extract tagged_recordings into pickle file                             
  pickle_file_path = f"/{dataset_path}/tagged_data.pkl"
  with open(pickle_file_path, 'wb') as file:
    pickle.dump(tagged_recordings, file)        

if __name__ == '__main__':
    # Receive input from user
    dataset_path = input("Enter path for dataset: ")
    lebel_data(dataset_path)

