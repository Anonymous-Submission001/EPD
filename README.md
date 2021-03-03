![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
# Euphemistic Phrase Detection by Masked Language Model
This repo is the Python 3 implementation of __Euphemistic Phrase Detection by Masked Language Model__.


## Table of Contents
- [Introduction](#Introduction)
- [Requirements](#Requirements)
- [Data](#Data)
- [Code](#Code)
- [Citation](#Citation)


## Introduction
Euphemisms, as an instrument to conceal secret information, have long been used on social media platforms. 
For instance, drug dealers often use "pot" for marijuana and "coke" for cocaine. 
This project aims at dicovering __euphemistic phrases__ that are used in a text corpus (e.g., a set of posts from an online forum). 
For instance, we aim to find "black tar" (heroin) and "blue dream" (marijuana) in a set of drug-related posts.



## Requirements
The code is based on Python 3.7. Please install the dependencies as below:  
```
pip install -r requirements.txt
```

## Data
Due to the license issue, we will not distribute the dataset ourselves, but we will direct the readers to their respective sources.  

__Drug__: 
- _Raw Text Corpus_: please request the dataset from [Zhu et al. (2021)] (Self-Supervised Euphemism Detection and
Identification for Content Moderation). 
- _Ground Truth_: we summarize the drug euphemism ground truth list (provided by the DEA Intelligence Report -- [Slang Terms and Code Words: A Reference for Law Enforcement Personnel](https://www.dea.gov/sites/default/files/2018-07/DIR-022-18.pdf)) in `data/answer_drug.txt`. 

__Sample__:
- _Raw Text Corpus_: we provide a sample dataset `data/sample.txt` for the readers to run the code.
- _Ground Truth_: same as the Drug dataset (see `data/answer_drug.txt`).  


## Code:
__1. Phrase Mining by AutoPhrase__:

Please refer to [this link](https://github.com/shangjingbo1226/AutoPhrase) to mine quality phrases from a text corpus. 
A sample of our generated phrases can be downloaded [here](https://drive.google.com/file/d/1aqvu5yGDUUtUr8MvChIBtvtdHQGp0_oA/view?usp=sharing). Please unzip it and put it under `data/phrase/`.


__2. Fine-tune the SpanBERT model__: 

Please refer to [this link](https://github.com/facebookresearch/SpanBERT) for model fine-tuning.
Our pre-trained model can be downloaded [here](https://drive.google.com/file/d/1pRPcbVNXIYbrsWzNNr1kO-7JEsDjGrWg/view?usp=sharing). Please unzip it and put it under `data/`. 


__3. Main Function__:

For a sample use of our code, please run the following command: 
```
python ./Main.py --target drug --dataset sample --model word2vec
```

You may specify the argument `model` to be one of `['epd', 'word2vec', 'epd-rank-all']`. 


## Citation
```bibtex
@inproceedings{anonymous2021euphemistic,
    title = {Euphemistic Phrase Detection by Masked Language Model},
    author = {Anonymous},
    booktitle = {The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    year = {2021}
}
```