from evaluate import load
bertscore = load("bertscore")

import pandas as pd
queries = pd.read_csv('./middle_products/queries_19.csv')
qids = queries.qid.tolist()
res = pd.read_csv('./res/bm25_dl_19.csv')

# for qid in queries.qid:
    # retrieved_num = res[res.qid==qid]['rank'].max()
    # print(retrieved_num)
    # print(res[(res.qid==qid)&(res['rank']<100)].head(10))
    
docnos = res.docno.unique().tolist()

import pickle

with open('../lab/MSMARCO_passage-QPP_experiments/msmarco_passage_dict.pkl', 'rb') as f:  # too big a file, stored in lab\MSMARCO_passage-QPP_experiments
    doc_dict = pickle.load(f)
    f.close()

new_doc_dict = {}
for docno in docnos:
    new_doc_dict.update({str(docno): doc_dict[str(docno)]})
    
with open('./middle_products/msmarco_passage_v1_retrieved_top_tail.pkl', 'wb') as f:
    pickle.dump(new_doc_dict, f)
    f.close()