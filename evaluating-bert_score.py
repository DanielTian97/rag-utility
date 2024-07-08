from evaluate import load
bertscore = load("bertscore")

import pandas as pd
queries = pd.read_csv('./middle_products/queries_19.csv')
qids = queries.qid.tolist()
qrels = pd.read_csv('./middle_products/qrels.csv')
docnos = str(qrels[(qrels.qid==qids[0])&(qrels.label==0)].docno.tolist()[0])

import pickle

with open('./middle_products/msmarco_passage_v1_qrels.pkl', 'rb') as f:
    doc_dict = pickle.load(f)

doc_text = doc_dict[docnos]
print(doc_text)
print(len(doc_dict))

pred_text = " Yes, goldfish can grow, but their growth rate depends on various factors such as water quality, diet, and living conditions. On average, a goldfish can grow up to 2-5 inches (5-13 cm) in length, but some breeds can grow larger."
predictions = [pred_text,]
references = [doc_text,]
results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")

print(results)