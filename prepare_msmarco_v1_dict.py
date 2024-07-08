from evaluate import load
bertscore = load("bertscore")

import pandas as pd
queries = pd.read_csv('./middle_products/queries_19.csv')
qids = queries.qid.tolist()
qrels = pd.read_csv('./middle_products/qrels.csv')

import pickle

with open('./middle_products/msmarco_passage_dict.pkl', 'rb') as f:  # too big a file, stored in lab\MSMARCO_passage-QPP_experiments
    doc_dict = pickle.load(f)

new_doc_dict = {}
for docno in qrels.docno.unique():
    new_doc_dict.update({str(docno): doc_dict[str(docno)]})
    
with open('./middle_products/msmarco_passage_v1_qrels.pkl', 'wb') as f:
    pickle.dump(new_doc_dict, f)