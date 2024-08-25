#!/bin/sh
python rag-random_kshot.py 1 1 5 10 10 0.3 20
echo 'Finished generation for 1shot for trec-dl-20!'
python evaluation_result_full.py 1 5 20
echo 'Finished bert-score evaluation for 1shot for trec-dl-20!'