#!/bin/sh
python rag-random_kshot.py 2 2 5 10 10 0.3 20
echo 'Finished generation for 2shot for trec-dl-20!'
python evaluation_result_full_20.py 2 5 20
echo 'Finished bert-score evaluation for 2shot for trec-dl-20!'

