#!/bin/sh
python rag-random_kshot.py 7 7 5 10 10 0.3 20
echo 'Finished generation for 7shot for trec-dl-20!'
python evaluation_result_full.py 7 5 20
echo 'Finished bert-score evaluation for 7shot for trec-dl-20!'

