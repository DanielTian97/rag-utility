import pandas as pd
import pickle

import pyterrier as pt
if not pt.started():
    pt.init()

import pandas as pd

dataset = pt.get_dataset('irds:msmarco-passage-v2')

res_21 = pd.read_csv('./res/bm25_dl_21.csv')
res_22 = pd.read_csv('./res/bm25_dl_22.csv')
qrels = pd.read_csv('./middle_products/qrels_v2.csv')

docnos_21 = res_21.docno.unique()
docnos_22 = res_22.docno.unique()
docnos_qrels = qrels.docno.unique()

print(type(docnos_21))
docnos = list(docnos_21)
docnos.extend(list(docnos_22))
docnos.extend(list(docnos_qrels))
print(len(docnos))

import pickle



try:
    filename = './middle_products/msmarco_passage_v2_dict_full.pkl'
    with open('./middle_products/msmarco_passage_v2_dict_full.pkl', 'rb') as f:
        full_doc_dict = pickle.load(f)
        f.close()
except:
    full_doc_dict = {}
    for i in dataset.get_corpus_iter(verbose=True):
        full_doc_dict.update({i['docno']: i['text']})

    with open('./middle_products/msmarco_passage_v2_dict_full.pkl', 'wb') as f:
        pickle.dump(full_doc_dict, f)
        f.close()

doc_dict = {}
for qid in full_doc_dict:
    doc_dict.update({qid: full_doc_dict[qid]})

with open('./middle_products/msmarco_passage_v2_dict.pkl', 'wb') as f:
    pickle.dump(doc_dict, f)
    f.close()