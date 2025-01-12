#!/bin/sh

python rag-random_kshot_prompt1.py 3 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 3shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 3 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 3shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 3 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 3shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 3 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 3shot for trec-dl-22!'

python rag-random_kshot_prompt1.py 4 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 4shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 4 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 4shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 4 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 4shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 4 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 4shot for trec-dl-22!'

python rag-random_kshot_prompt1.py 6 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 6shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 6 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 6shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 6 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 6shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 6 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 6shot for trec-dl-22!'

python rag-random_kshot_prompt1.py 7 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 7shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 7 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 7shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 7 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 7shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 7 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 7shot for trec-dl-22!'



