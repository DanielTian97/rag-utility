#!/bin/sh
python rag-permutation-random_kshot.py 3 3 5 5 0 0.3
echo 'Finished generation for 3shot for trec-dl-19 (permutation)!'
python evaluation_result.py 3 5 p
echo 'Finished bert-score evaluation for 3shot for trec-dl-19 (permutation)!'