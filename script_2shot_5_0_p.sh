#!/bin/sh
python rag-permutation-random_kshot.py 2 2 5 5 0 0.3
echo 'Finished generation for 2shot for trec-dl-19 (permutation)!'
python evaluation_result.py 2 5 p
echo 'Finished bert-score evaluation for 2shot for trec-dl-19 (permutation)!'