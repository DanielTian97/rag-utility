#!/bin/sh
python rag-permutation-random_kshot_prompt1.py 2 1 5 1 0 0.3 False 19
echo 'Finished generation for 2shot(new prompt) 4 permutations for trec-dl-19!'
python rag-permutation-random_kshot_prompt1.py 3 1 5 1 0 0.3 False 19
echo 'Finished generation for 3shot(new prompt) 4 permutations for trec-dl-19!'
python rag-permutation-random_kshot_prompt1.py 4 1 5 1 0 0.3 False 19
echo 'Finished generation for 4shot(new prompt) 4 permutations for trec-dl-19!'
python rag-permutation-random_kshot_prompt1.py 5 1 5 1 0 0.3 False 19
echo 'Finished generation for 5shot(new prompt) 4 permutations for trec-dl-19!'