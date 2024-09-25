#!/bin/sh
python rag-random_kshot_prompt1.py 1 1 5 10 0 0.3 19
echo 'Finished generation for 1shot(new prompt) for trec-dl-19!'
python rag-random_kshot_prompt1.py 2 1 5 9 0 0.3 19
echo 'Finished generation for 2shot(new prompt) for trec-dl-19!'
python rag-random_kshot_prompt1.py 3 1 5 8 0 0.3 19
echo 'Finished generation for 3shot(new prompt) for trec-dl-19!'
python rag-random_kshot_prompt1.py 4 1 5 7 0 0.3 19
echo 'Finished generation for 4shot(new prompt) for trec-dl-19!'
python rag-random_kshot_prompt1.py 5 1 5 6 0 0.3 19
echo 'Finished generation for 5shot(new prompt) for trec-dl-19!'