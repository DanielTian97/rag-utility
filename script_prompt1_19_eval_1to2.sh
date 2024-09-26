#!/bin/sh
python evaluation_result_full_new_prompt.py 1 5 19 10 0
echo 'Finished bert-score evaluation for 1shot for trec-dl-19(new prompt)!'
python evaluation_result_full_new_prompt.py 2 5 19 9 0
echo 'Finished bert-score evaluation for 2shot for trec-dl-19(new prompt)!'