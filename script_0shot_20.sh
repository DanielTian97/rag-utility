#!/bin/sh
python rag-random-0shot.py 5 20
echo 'Finished generation for 0shot for trec-dl-20!'
python evaluation_result_full.py 0 5 20
echo 'Finished bert-score evaluation for 0shot for trec-dl-20!'