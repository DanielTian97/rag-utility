#!/bin/sh
python rag-random_kshot.py 5 5 5 10 10 0.3 20
echo 'Finished generation for 5shot for trec-dl-20!'
python evaluation_result_full.py 5 5 20
echo 'Finished bert-score evaluation for 5shot for trec-dl-20!'

