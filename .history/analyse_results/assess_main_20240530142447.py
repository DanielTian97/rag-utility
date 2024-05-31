import os
import sys
from upstream_assess import *
from downstream_assess import *

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if __name__=="__main__":
    dataset = str(sys.argv[1]) # dataset = 'msmarco_passage' or 'msmarco_passage_v2'
    rerank_cutoff = int(sys.argv[2]) # <=100
    coefficient = int(sys.argv[3])

print(os.getcwd())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' #disenable tensorflow messages

current = os.path.dirname(os.path.realpath(__file__))
print(current)

parent = os.path.dirname(current)
print(parent)
sys.path.append(parent)

# dataset = 'msmarco_passage'

if('analyse_results' in os.getcwd()):
    file_path = f'./prediction_record_{dataset}.pkl'
    res_folder_path = f'../eva/'
    mp_folder_path = f'../middle_products/'
else:
    file_path = f'./analyse_results/prediction_record_{dataset}.pkl'
    res_folder_path = f'./eva/'
    mp_folder_path = f'./middle_products/'

from compose_prompts import *
print(file_path)
with open(file_path, 'rb') as f:
    import pickle
    result = pickle.load(f)
    f.close()
    
import pandas as pd
if('v2' not in 'dataset'):
    eva_0 = pd.read_csv(res_folder_path+'bm25_dl_19.eva.csv')
    eva_1 = pd.read_csv(res_folder_path+'bm25_dl_20.eva.csv')
    pred_eva = pd.read_csv(res_folder_path+f'pred_eva_bm25_dl_19_20-Q{coefficient}.csv')
    queries_0 = pd.read_csv(mp_folder_path+'queries_19.csv')['qid'].astype('str').values
    queries_1 = pd.read_csv(mp_folder_path+'queries_20.csv')['qid'].astype('str').values
else:
    eva_0 = pd.read_csv(res_folder_path+'bm25_dl_21.eva.csv')
    eva_1 = pd.read_csv(res_folder_path+'bm25_dl_22.eva.csv')
    pred_eva = pd.read_csv(res_folder_path+f'pred_eva_bm25_dl_21_22-Q{coefficient}.csv')
    queries_0 = pd.read_csv(mp_folder_path+'queries_21.csv')['qid'].astype('str').values
    queries_1 = pd.read_csv(mp_folder_path+'queries_22.csv')['qid'].astype('str').values
eva = pd.concat([eva_0, eva_1])
eva.qid = eva.qid.astype('str')

pred_eva.qid = pred_eva.qid.astype('str')

# # quantification
# coefficient = 100
# for pair in result:
#     pair.pred = int(pair.pred*coefficient)

# pseudo_qrel = []
# for record in result:
#     pseudo_qrel.append([record.qid, record.docno, record.pred, 0])
    
# import pandas as pd
# pseudo_qrel_df = pd.DataFrame(pseudo_qrel, columns=['qid', 'docno', 'label', 'iteration'])
# print(pseudo_qrel_df.head())

# if(coefficient<1):
#     coefficient = -int(1/coefficient)
# pseudo_qrel_df.to_csv(f'../middle_products/psuedo_qrel_{dataset}-Q{coefficient}.csv', index=False)
        
ndcg_pred_qrel(result, rerank_cutoff=rerank_cutoff)
correlation(result, eva, pred_eva, [queries_0, queries_1])