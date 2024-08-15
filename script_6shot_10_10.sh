#!/bin/sh
python rag-baseline-random_kshot.py 6 6 5 10 10 0.3
echo 'Finished generation for 4shot for trec-dl-19!'
python evaluation_result.py 6 5
echo 'Finished bert-score evaluation for 6shot for trec-dl-19!'