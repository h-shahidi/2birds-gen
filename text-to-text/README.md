# Neural Question Generation
This repository provides implementation for neural question generation, discussed in our paper https://arxiv.org/pdf/1909.10158.pdf.

# Dependencies
  - Python 2.7
  - Tensorflow 1.10.0
  - nltk
  - numpy
  - stanfordcorenlp

# Data
The [SQuAD1.1](https://rajpurkar.github.io/SQuAD-explorer/) dataset is used for training and evaluation. Prior work has splitted the data into train, validation, and test sets in two different ways, which we will call split-1 and split-2, respectively.

We provide our data split-1 [here](https://drive.google.com/file/d/1Avd7EBY7652r09UeIxngn8yiPHbX9qB6/view?usp=sharing). Download and unzip it inside `data/split-1` directory or move to `Preprocess Data Split-1` section, if you'd like to prepare it yourself.

For data split-2, download and unzip `QG` directory from [here](https://res.qyzhou.me/redistribute.zip) inside `data/split-2` directory. Then, run the following command:
```
bash rename_files.sh
```

We use the vocabulary and word embedding that is provided by [MPQG](https://github.com/freesunshine0316/MPQG).

# Preprocess Data Split-1
To prepare data split-1 yourself, download the data provided by [MPQG](https://github.com/freesunshine0316/MPQG) from [here](https://www.cs.rochester.edu/~lsong10/downloads/nqg_data.tgz) and unzip it inside `data/split-1` directory. Then, download the raw SQuAD dataset from [here](https://github.com/xinyadu/nqg/tree/master/data/raw) in the same directory. In addition, we would need [Stanford CoreNLP](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip) for tokenization, POS tagging, etc. Eventually, run the following command in `data/split-1` directory:
```
python preprocess.py
```

# Train
Run the following command for training:
```
bash scripts/train.sh
```

If you want to train the model using reinforcement learning change the mode in `config.json` to `rl_train`. The model and logs would be saved at `logs`.

# Test
Run the following command for generating questions using beam search:
```
bash scripts/generate.sh
```

Then, run the following command to evaluate the obtained results and get the scores.
```
bash scripts/evaluate.sh
```

For evaluation, the metrics are computed by the same scripts as in [nqg](https://github.com/xinyadu/nqg). These scripts are adopted from [coco-caption](https://github.com/tylin/coco-caption) repository.

# Acknowledgement
This code is based on [MPQG](https://github.com/freesunshine0316/MPQG). We would like to thank the authors for sharing their code base.