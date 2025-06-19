# How Does a Deep Neural Network Look at Lexical Stress?

<h1 align="center">
  <br>
[JASA] How Does a Deep Neural Network Look at Lexical Stress?: Minimal-Pairs Lexical-Stress Classification with LRP Analysis
  <br>
  <img src="https://github.com/ItaiAllouche/minimalPairsLexicalStress/blob/main/figs/spcts_and_heatmaps.png" height="370">
</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/itai-allouche/">Itai Allouche</a> •
    <a href="https://il.linkedin.com/in/itay-asael-37769322a?original_referer=https%3A%2F%2Fwww.google.com%2F">Itay Asael</a> •
    <a href="https://il.linkedin.com/in/rotem-rousso-897444213">Rotem Rousso</a> •
    <a href="https://il.linkedin.com/in/vered-dassa-202722185">Vered Dassa</a> •
    <a href="https://www.linkedin.com/in/ann-bradlow-8344939">Ann Bradlow</a> •
    <a href="https://faculty.wcas.northwestern.edu/matt-goldrick/#!/">Matt Goldrick </a> •    
    <a href="https://il.linkedin.com/in/jkeshet">Yossi Keshet</a> •
  </p>
  
  <h2 align="center">
  Official repository of the paper
  </h2>

***How Does a Deep Neural Network Look at Lexical Stress?***

*Itai Allouche, Itay Asael, Rotem Rousso, Vered Dassa, Ann Bradlow Matt, Goldrick and Yossi Keshet*

***Abstract**: Deep learning architectures are highly impressive on many speech processing tasks; what
underlies such improvements? This work examines this general issue in the context of lexical
stress. A dataset of English disyllabic words was automatically constructed from read and
spontaneous speech. Convolutional Neural Network (CNNs) were trained to predict stress
position from a spectrographic representation of the acoustics of non-minimal pairs (initial
WAllet vs. final exTEND), achieving up to 92% accuracy on held-out test data. Layerwise
Relevance Propagation (LRP) analysis revealed that the CNNs’ predictions for held-out
minimal pairs (PROtest vs. proTEST )) were most strongly influenced by information in
stressed vs. unstressed syllables, in particular stressed vowels spectral properties. However,
the classifiers also attended to information throughout the word. A feature-specific relevance
analysis is proposed; its results suggest our best-performing classifier is strongly influenced
by the stressed vowel’s first and second formant, with some evidence that its pitch and
third formant contribute as well. These results reveal deep learning’s ability to acquire
distributed cues to stress from naturally occurring data, extending traditional phonetic work
based around highly controlled stimuli.*

---

- [How Does a Deep Neural Network Look at Lexical Stress?](#What-Does-Lexical-Stress-Look-Like)
- [A reproducible demo of lexical-stress classification and LRP-based interpretation](#A-reproducible-demo-of-lexical-stress-classification-and-LRP-based-interpretation)
  * [TheRepository Oganization](#Repository-Oganization)
  * [Installation](#Installation)
  * [Run Demo](#Run-Demo)
  * [TODO](#TODO)
## A Reproducible Demo Of Lexical Stress Classification and LRP-based Interpretation
### Repository Oganization
| File | Purpose |
|---------|------------------|
| [`examples/`](examples/) | Demo audio clips (`.wav`). `IS/` and `FS/` sub-folders |
| [`checkpoints/`](checkpoints/) | Checkpoints of trained models (currently only VGG16 is supported) |
| [`model.py`](model.py)   | VGG variants and ResNet-18 architectures (PyTorch) |
| [`utils/lrp.py`](utils/lrp.py)           | Composite LRP rules implemented with **Captum** |
| [`utils/loader.py`](utils/loader.py)           | Spectogram and Model loader methods |
| [`run_demo.ipynb`](run_demo.ipynb) | One-click notebook: load model → classify → visualise LRP |
| [`requirements.txt`](requirements.txt) |Python dependencies|


### Installation

```bash
# clone the project
git clone https://github.com/ItaiAllouche/minimalPairsLexicalStress.git
cd minimalPairsLexicalStress

# create and activate virtual environment (recommended)
python -m venv lexicalStress
source lexicalStress/bin/activate # windows: lexicalStress\Scripts\activate

# install the required packages
pip install -r requirements.txt
```
This commands clones the project, creates & activates a python virtual environment, and install all the relevant python dependencies.
### Run Demo
Run the jupyter notebook `run_demo.ipynb` 

### TODO
Add ResNet architecture to this demo

