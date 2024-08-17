#!/bin/sh
python rag-baseline-random_kshot.py 3 3 5 10 10 0.3
echo 'Finished generation for 3shot for trec-dl-19!'
python evaluation_result.py 3 5
echo 'Finished bert-score evaluation for 3shot for trec-dl-19!'