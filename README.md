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
| [`examples/`](examples/) | Demo audio clips (`.wav`). `IS/` and `FS/` sub-folders |
| [`model.py`](model.py)   | VGG variants and ResNet-18 architectures (PyTorch) |
| [`utils/lrp.py`](utils/lrp.py)           | Composite LRP rules implemented with **Captum** |
| [`utils/loader.py`](utils/loader.py)           | Spectogram and Model loader methods |
| [`run_demo.ipynb`](run_demo.ipynb) | One-click notebook: load model → classify → visualise LRP |
| [`requirements.txt`](requirements.txt) |Python dependencies|

---

## Installation

```bash
# 1 Create virtual environment (recommended)
python -m venv lexicalStress
source lexicalStress/bin/activate     # Windows: lexicalStress\Scripts\activate

# 2 Install the required packages
pip install -r requirements.txt
```
This commands create and activate a python virtual environment, and install all the relevant python dependencies.
## Run Demo
Run the jupyter notebook `run_demo.ipynb` 

