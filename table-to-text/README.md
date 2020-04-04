# Neural Table-To-Text

## Dependencies
  - Python 2.7
  - Tensorflow 1.0.0
  - nltk
  - matplotlib

## Data
The [WikiBio](https://github.com/DavidGrangier/wikipedia-biography-dataset) dataset is used for training and evaluation. The dataset has been divided into training (80%), validation (10%), and test (10%) sets.

[wiki2bio](https://github.com/tyliupku/wiki2bio) has processed the dataset and provided [`original_data`](https://drive.google.com/file/d/15AV8LeWY3nzCKb8RRbM8kwHAp_DUZ5gf/view?usp=sharing). In the `original_data`, `*.box` are the infoboxes, and `*.summary` are the corresponding biographies. `word_vocab.txt` and `field_vocab.txt` are word and field vocabularies, respectively. 

## Preprocess
Run the following command:
```
python preprocess.py
```

After preprocessing, `processed_data` would be created. This directory contains `*.box.pos`, `*.box.rpos`, `*.box.val`, and `*.box.lab` that represent the word position p+, p-, field content and field types, respectively.

## Train
Run the following command for training:
```
python main.py --mode=train
```

If you want to train the model using reinforcement learning run the following command:
```
python main.py --mode=train --loss=rl
```

Experimental results will be saved at `results/res`.

## Test
Run the following command for testing:
```
python main.py --mode=test
```