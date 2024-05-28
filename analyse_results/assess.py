import os
import sys
print(os.getcwd())

current = os.path.dirname(os.path.realpath(__file__))
print(current)

parent = os.path.dirname(current)
print(parent)
sys.path.append(parent)

if('analyse_results' in os.getcwd()):
    file_path = './prediction_record_msmarco_passage.pkl'
else:
    file_path = './analyse_results/prediction_record_msmarco_passage.pkl'

from compose_prompts import *
print(file_path)
with open(file_path, 'rb') as f:
    import pickle
    result = pickle.load(f)
    f.close()

list_score = {-1: [], 0: [], 1: [], 2: [], 3: []}
for record in result:
    list_score[record.qrel].append(record.pred)

pred_list = []
qrel_list = []
for i in list_score:
    print(i, sum(list_score[i])/len(list_score[i]), min(list_score[i]), max(list_score[i]))
    for v in list_score[i]:
        pred_list.append(v)
        qrel_list.append(max(0, min(i, 2)))

import tensorflow_ranking as tfr

y_true = [qrel_list]
y_pred = [pred_list]
ndcg = tfr.keras.metrics.NDCGMetric(topn=100)
print(ndcg(y_true, y_pred).numpy())