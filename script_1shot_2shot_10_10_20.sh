#!/bin/sh
python rag-random_kshot.py 1 1 5 10 10 0.3 20
echo 'Finished generation for 1shot for trec-dl-20!'
python rag-random_kshot.py 2 2 5 10 10 0.3 20
echo 'Finished generation for 2shot for trec-dl-20!'