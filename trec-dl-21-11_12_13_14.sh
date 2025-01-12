#!/bin/sh

python rag-random_kshot_prompt1.py 14 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 14shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 14 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 14shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 14 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 14shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 14 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 14shot for trec-dl-22!'

python rag-random_kshot_prompt1.py 13 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 13shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 13 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 13shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 13 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 13shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 13 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 13shot for trec-dl-22!'

python rag-random_kshot_prompt1.py 12 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 12shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 12 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 12shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 12 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 12shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 12 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 12shot for trec-dl-22!'

python rag-random_kshot_prompt1.py 11 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 11shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 11 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 11shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 11 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 11shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 11 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 11shot for trec-dl-22!'



