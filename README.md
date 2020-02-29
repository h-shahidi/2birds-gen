# Neural Question Generation
This repository provides implementation for neural question generation, discussed in our paper https://arxiv.org/pdf/1909.10158.pdf.

# Dependencies
  - Python 2.7
  - Tensorflow 1.10.0
  - nltk
  - numpy
  - stanfordcorenlp

# Data
The [SQuAD1.1](https://rajpurkar.github.io/SQuAD-explorer/) dataset is used for training and evaluation. We provide our data through `git-lfs`. [This page](https://github.com/git-lfs/git-lfs/wiki/Installation) explains how to install it. You should be able to get the data in `data` directory using a regular git clone.

# Preprocess
To prepare data split-1 yourself, download the data provided by [MPQG](https://github.com/freesunshine0316/MPQG) from[here](https://www.cs.rochester.edu/~lsong10/downloads/nqg_data.tgz) and unzip it inside `data` directory. Then, download the raw SQuAD dataset from [here](https://github.com/xinyadu/nqg/tree/master/data/raw) in the same directory. In addition, we would need [Stanford CoreNLP](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip) for tokenization, POS tagging, etc. Eventually, run the following command in `data` directory:
```
python preprocess.py
```

# Train
Run the following command for training:
```
source scripts/train.sh
```

If you want to train the model using reinforcement learning change the mode in `config.json` to `rl_train`. The model and logs would be saved at `logs`.

# Test
Run the following command for generating questions using beam search:
```
source scripts/generate.sh
```

Then, run the following command to evaluate the obtained results and get the scores.
```
source scripts/evaluate.sh
```

For evaluation, the metrics are computed by the same scripts as in [nqg](https://github.com/xinyadu/nqg) has used. These scripts are adopted from [coco-caption](https://github.com/tylin/coco-caption) repository.

# Acknowledgement
This code is based on [MPQG](https://github.com/freesunshine0316/MPQG). We would like to thank the authors for sharing their code base.