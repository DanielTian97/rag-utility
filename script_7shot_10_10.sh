#!/bin/sh
python rag-baseline-random_kshot.py 7 7 5 10 10 0.3
echo 'Finished generation for 4shot for trec-dl-19!'
python evaluation_result.py 7 5
echo 'Finished bert-score evaluation for 6shot for trec-dl-19!'