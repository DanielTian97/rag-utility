#!/bin/sh
python rag-permutation-random_kshot_prompt1.py 2 1 5 1 0 0.3 False 19 mt5
echo 'Finished generation for 6shot(new prompt) 2 permutations for trec-dl-19 mt5!'
python rag-permutation-random_kshot_prompt1.py 3 1 5 1 0 0.3 False 19 mt5
echo 'Finished generation for 7shot(new prompt) 3 permutations for trec-dl-19 mt5!'
python rag-permutation-random_kshot_prompt1.py 4 1 5 1 0 0.3 False 19 mt5
echo 'Finished generation for 8shot(new prompt) 4 permutations for trec-dl-19 mt5!'
python rag-permutation-random_kshot_prompt1.py 5 1 5 1 0 0.3 False 19 mt5
echo 'Finished generation for 8shot(new prompt) 5 permutations for trec-dl-19 mt5!'