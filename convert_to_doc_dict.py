import pandas as pd
import pickle
import sys

# #we are using pyterrier's iter_col instead of the function in this file
# def make_pkl(collection_path: str):
#     # msmarco_passage: '../lab/collection/collection.tsv'
#     # msmarco_passage_v2: '../lab/collection/collection_v2.tsv'
#     document_df = pd.read_csv(collection_path, sep='\t', names=['docno', 'text'], encoding='windows-1252')
#     document_df.docno = document_df.docno.astype('str')
#     doc_dict = dict(zip(document_df.docno, document_df.text))
#     print(len(doc_dict))

#     # with open('./middle_products/msmarco_passage.pkl', 'wb') as f:
#     #     pickle.dump(doc_dict, f)
#     #     f.close()
#     print(document_df[document_df.docno=='8182161']['text'].to_list()[0])
        
# if __name__=="__main__":
#     collection_path = str(sys.argv[1])
#     make_pkl(collection_path)

import pyterrier as pt
if not pt.started():
    pt.init()

import pandas as pd

res_2021 = pd.read_csv('./res/bm25_dl_21.csv')
res_2022 = pd.read_csv('./res/bm25_dl_22.csv')

docno_set = set(res_2021.docno).union(set(res_2021.docno))

dataset_2021 = pt.get_dataset('irds:msmarco-passage-v2/trec-dl-2021/judged')

import pickle

doc_dict = {}
for i in dataset_2021.get_corpus_iter(verbose=True):
    if(i['docno'] in docno_set):
        doc_dict.update({i['docno']: i['text']})

with open('./middle_products/msmarco_passage_v2.pkl', 'wb') as f:
    pickle.dump(doc_dict, f)
        
# with open('./middle_products/msmarco_passage_v2.pkl', 'rb') as f:
#     doc_dict = pickle.load(f)