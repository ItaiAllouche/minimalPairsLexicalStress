# What Does Lexical Stress Look Like?

*A reproducible demo of lexical-stress classification and LRP-based interpretation*

This repository accompanies the JASA paper **“ What Does Lexical Stress Look Like?”**  
It contains everything you need to:

1. **Classify** disyllabic English words as **initial-stress (IS)** or **final-stress (FS)** with a VGG-16 CNN.
2. **Interpret** the model’s decisions via **Layer-wise Relevance Propagation (LRP)** heat-maps.

---

## Table of contents
| File | Purpsoe |
|---------|------------------|
| [`examples/`](examples/) | Demo audio clips (`.wav`) &mdash; `IS/` and `FS/` sub-folders |
| [`model.py`](model.py)   | VGG variants and ResNet-18 architectures (PyTorch) |
| [`lrp/`](lrp/)           | Composite LRP rules implemented with **Captum** |
| [`run_demo.ipynb`](run_demo.ipynb) | One-click notebook: load model ▶ classify ▶ visualise LRP |
| [`requirements.txt`](requirements.txt) ||

---

## Installation

```bash
# 1 Create virtual environment (recommended)
python -m venv lexicalStress
source lexicalStress/bin/activate     # Windows: lexicalStress\Scripts\activate

# 2 Install the required packages
pip install -r requirements.txt
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
