#!/bin/sh

python rag-random_kshot_prompt1.py 9 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 9shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 9 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 9shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 9 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 9shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 9 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 9shot for trec-dl-22!'

python rag-random_kshot_prompt1.py 8 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 8shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 8 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 8shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 8 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 8shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 8 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 8shot for trec-dl-22!'

python rag-random_kshot_prompt1.py 1 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 1shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 1 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 1shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 1 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 1shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 1 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 1shot for trec-dl-22!'

python rag-random_0shot_prompt1.py 0 0 5 0 0 0.3 21 bm25
echo 'Finished generation for 0shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 0 5 21 0 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 0shot for trec-dl-21!'
python rag-random_0shot_prompt1.py 0 0 5 0 0 0.3 22 bm25
echo 'Finished generation for 0shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 0 5 22 0 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 0shot for trec-dl-22!'



