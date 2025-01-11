import pandas as pd
import pickle

import pyterrier as pt
if not pt.started():
    pt.init()

import pandas as pd

dataset = pt.get_dataset('irds:msmarco-passage-2')

import pickle

doc_dict = {}
for i in dataset.get_corpus_iter(verbose=True):
    doc_dict.update({i['docno']: i['text']})

with open('./middle_products/msmarco_passage_v2_dict.pkl', 'wb') as f:
    pickle.dump(doc_dict, f)