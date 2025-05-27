# What Does Lexical Stress Look Like?

<h1 align="center">
  <br>
[JASA] What Does Lexical Stress Look Like?: Minimal-Pairs Lexical-Stress Classification with LRP Analysis
  <br>
  <img src="https://github.com/ItaiAllouche/minimalPairsLexicalStress/blob/main/figs/spectogram_and_lrp_heatmaps.png" height="400">
</h1>
  <p align="center">
    <a href="https://github.com/ItaiAllouche">Itai Allouche</a> •
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

***What Does Lexical Stress Look Like?***

*Itai Allouche, Itay Asael, Rotem Rousso, Vered Dassa, Ann Bradlow Matt Goldrick and Yossi Keshet*

***Abstract**: Lexical stress plays a crucial role in distinguishing word meanings and grammatical functions,
particularly in minimal pairs (e.g., PREsent vs. presENT ). The aim is to train a classifier
for detecting the stressed syllable on a large amount of data, and understanding the acoustic
features underlying its decisions. Disyllabic stress minimal word pairs and non-minimal word
pairs (e.g., WALlet vs. extEND) were extracted from multiple speech corpora using forced
alignment. A part-of-speech tagging system was used to label each minimal pairs word as ei-
ther a noun, which is associated with stress on the first syllable, or a verb, which is associated
with stress on the last syllable. In non-minimal pairs, stress placement is unambiguous and
consistently follows lexical conventions. Several Convolutional neural network (CNN) archi-
tectures were trained using focal loss to mitigate class imbalance, with the best-performing
model achieving a classification accuracy of 92%. To interpret model behavior, Layerwise
Relevance Propagation (LRP) was applied, producing spectrogram heatmaps that highlight
key signal regions influencing classification. Additionally, acoustic features, such as funda-
mental frequency and the first three formants, were extracted and analyzed to assess their
contributions to the model’s predictions.*

---

- [What Does Lexical Stress Look Like?](#What-Does-Lexical-Stress-Look-Like\?)
- [A reproducible demo of lexical-stress classification and LRP-based interpretation](#A-reproducible-demo-of-lexical-stress-classification-and-LRP-based-interpretation)
  * [TheRepository Oganization](#Repository-Oganization)
  * [Installation](#Installation)
  * [Run Demo](#Run-Demo)
## A reproducible Demo Of Lexical Stress Classification and LRP-based Interpretation
### Repository Oganization
| File | Purpose |
|---------|------------------|
| [`examples/`](examples/) | Demo audio clips (`.wav`). `IS/` and `FS/` sub-folders |
| [`model.py`](model.py)   | VGG variants and ResNet-18 architectures (PyTorch) |
| [`utils/lrp.py`](utils/lrp.py)           | Composite LRP rules implemented with **Captum** |
| [`utils/loader.py`](utils/loader.py)           | Spectogram and Model loader methods |
| [`run_demo.ipynb`](run_demo.ipynb) | One-click notebook: load model → classify → visualise LRP |
| [`requirements.txt`](requirements.txt) |Python dependencies|

---

### Installation

```bash
# 1 Create virtual environment (recommended)
python -m venv lexicalStress
source lexicalStress/bin/activate     # Windows: lexicalStress\Scripts\activate

# 2 Install the required packages
pip install -r requirements.txt
```
This commands create and activate a python virtual environment, and install all the relevant python dependencies.
### Run Demo
Run the jupyter notebook `run_demo.ipynb` 

