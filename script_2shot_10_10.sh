#!/bin/sh
python rag-baseline-random_kshot.py 2 2 5 10 10 0.3
echo 'Finished generation for 2shot for trec-dl-19!'
python evaluation_result.py 2 5
echo 'Finished bert-score evaluation for 2shot for trec-dl-19!'