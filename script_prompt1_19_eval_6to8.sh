#!/bin/sh
python evaluation_result_full_new_prompt.py 6 5 19 5 0
echo 'Finished bert-score evaluation for 6shot for trec-dl-19(new prompt)!'
python evaluation_result_full_new_prompt.py 7 5 19 4 0
echo 'Finished bert-score evaluation for 7shot for trec-dl-19(new prompt)!'
python evaluation_result_full_new_prompt.py 8 5 19 3 0
echo 'Finished bert-score evaluation for 8shot for trec-dl-19(new prompt)!'