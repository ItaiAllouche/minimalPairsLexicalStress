# What Does Lexical Stress Look Like?

*A reproducible demo of lexical-stress classification and LRP-based interpretation*

This repository accompanies the JASA paper **“ What Does Lexical Stress Look Like?”**  
It contains everything you need to

1. **Classify** disyllabic English words as **initial-stress (IS)** or **final-stress (FS)** with a VGG-16 CNN, and  
2. **Interpret** the model’s decisions via **Layer-wise Relevance Propagation (LRP)** heat-maps.

---

## Table of contents
| Section | What you’ll find |
|---------|------------------|
| [`examples/`](examples/) | Demo audio clips (`.wav`) &mdash; `IS/` and `FS/` sub-folders |
| [`model.py`](model.py)   | VGG variants and ResNet-18 architectures (PyTorch) |
| [`lrp/`](lrp/)           | Composite LRP rules implemented with **Captum** |
| [`run_demo.ipynb`](run_demo.ipynb) | One-click notebook: load model ▶ classify ▶ visualise LRP |
| [`requirements.txt`](requirements.txt) | Python dependencies (PyTorch ≥1.12, Captum, Librosa, NumPy, etc.) |

---

## Installation

```bash
# 1 Create a fresh environment (recommended)
python -m venv venv          # or conda create -n lexicalstress python=3.9
source venv/bin/activate     # Windows: venv\Scripts\activate

# 2 Install the required packages
pip install -r requirements.txt
77% accuracy on test set
<br>

<img src="/img/30sec_test.jpeg">

### 15s model
<br>
The model was trained on 15s long tracks.
Each 30s track was divided into 2 sub-tracks 15s long
<br>
performance:
<br>
78.85% accuracy on validation set
<br>

<img src="/img/15sec_valid.jpeg">
<br>

75.5% accuracy on test set
<br>

<img src="/img/15sec_test.jpeg">

### 10s model
The model was trained on 10s tracks.
Each 30s track was divided into 3 sub-tracks 10s long
<br>
performance:

<br>
78% accuracy on validation set
<br>

<img src="/img/10sec_valid.jpeg">
<br>

74.5% accuracy on test set
<br>
<img src="/img/10sec_test.jpeg">
<br>
## Docker
The project is intended to run in huggingface docker image
<br>
For instructions on how to install docker:
<br>
<a href="https://docs.docker.com/engine/install/">https://docs.docker.com/engine/install/</a>
## Training
### Train 30s model
Replace `train_30s_model.py` with your chosen model
```bash
docker run --name gtzan --rm -it --ipc=host --gpus=all -v $PWD:/home huggingface/transformers-pytorch-gpu python3 /home/train_30s_model.py
```
This command spins up a docker container from the official huggingface image, mounts the repo directory and run the training script
## Running
### Run the model - from huggingface 🤗
Open the <a href="https://huggingface.co/adamkatav/wav2vec2_100k_gtzan_30s_model">Model</a> in hugging face.
<br>
<img src="/img/run_in_hugging_face.jpeg">
<br>
*Note that hugging face server supports tracks up to 2-3 minutes*
### Run the model - using python
#### On GPU:
```bash
docker run --name gtzan --rm -it --ipc=host --gpus=all -v $PWD:/home huggingface/transformers-pytorch-gpu
```
#### On CPU:
```bash
docker run --name gtzan --rm -it -v $PWD:/home huggingface/transformers-pytorch-gpu
```
In the container either use a python script file or via the interactive interpreter:
```python
from transformers import pipeline
import torchaudio
import sys
MODEL_NAME = 'adamkatav/wav2vec2_100k_gtzan_30s_model'
SONG_IN_REPO_DIR_PATH = '/home/rolling_stones.wav'

pipe = pipeline(model=MODEL_NAME)
audio_array,sample_freq = torchaudio.load(SONG_IN_REPO_DIR_PATH)
resample = torchaudio.transforms.Resample(orig_freq=sample_freq)
audio_array = audio_array.mean(axis=0).squeeze().numpy()
output = pipe(audio_array)
print(output)
```
