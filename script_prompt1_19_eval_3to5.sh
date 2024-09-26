#!/bin/sh
python evaluation_result_full_new_prompt.py 3 5 19 8 0
echo 'Finished bert-score evaluation for 3shot for trec-dl-19(new prompt)!'
python evaluation_result_full_new_prompt.py 4 5 19 7 0
echo 'Finished bert-score evaluation for 4shot for trec-dl-19(new prompt)!'
python evaluation_result_full_new_prompt.py 5 5 19 6 0
echo 'Finished bert-score evaluation for 5shot for trec-dl-19(new prompt)!'