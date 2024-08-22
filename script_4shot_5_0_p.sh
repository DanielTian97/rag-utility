#!/bin/sh
python rag-permutation-random_kshot.py 4 4 5 5 0 0.3
echo 'Finished generation for 4shot for trec-dl-19 (permutation)!'
python evaluation_result.py 4 5 p
echo 'Finished bert-score evaluation for 4shot for trec-dl-19 (permutation)!'