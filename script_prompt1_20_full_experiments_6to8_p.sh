#!/bin/sh
python rag-permutation-random_kshot_prompt1.py 6 1 5 1 0 0.3 False 20
echo 'Finished generation for 6shot(new prompt) 4 permutations for trec-dl-20!'
python rag-permutation-random_kshot_prompt1.py 7 1 5 1 0 0.3 False 20
echo 'Finished generation for 7shot(new prompt) 4 permutations for trec-dl-20!'
python rag-permutation-random_kshot_prompt1.py 8 1 5 1 0 0.3 False 20
echo 'Finished generation for 8shot(new prompt) 4 permutations for trec-dl-20!'