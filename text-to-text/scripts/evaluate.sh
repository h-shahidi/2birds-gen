#!/bin/bash
python utils/remove_duplicates.py --source_file results/output.pred \
                                  --out_file results/processed_output.pred \
                                  --ngram 4
python qgevalcap/eval.py --tgt_file results/output.ref \
                         --out_file results/processed_output.pred
