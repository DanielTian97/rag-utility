import os
import sys
from upstream_assess import *
print(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #disenable tensorflow messages

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
    
ndcg_pred_qrel(result, 10)