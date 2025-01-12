#!/bin/sh
python rag-random_kshot_prompt1.py 2 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 2shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 2 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 2shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 5 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 5shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 5 5 21 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 5shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 10 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 10shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 10 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 10shot for trec-dl-21!'
python rag-random_kshot_prompt1.py 15 1 5 1 0 0.3 21 bm25
echo 'Finished generation for 15shot for trec-dl-21!'
python evaluation_result_full_new_prompt.py 15 5 21 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 15shot for trec-dl-21!'

python rag-random_kshot_prompt1.py 2 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 2shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 2 5 22 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 2shot for trec-dl-22!'
python rag-random_kshot_prompt1.py 5 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 5shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 5 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 5shot for trec-dl-22!'
python rag-random_kshot_prompt1.py 10 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 10shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 10 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 10shot for trec-dl-22!'
python rag-random_kshot_prompt1.py 15 1 5 1 0 0.3 22 bm25
echo 'Finished generation for 15shot for trec-dl-22!'
python evaluation_result_full_new_prompt.py 15 5 22 1 0 _prompt1 bm25
echo 'Finished bert-score evaluation for 15shot for trec-dl-22!'

