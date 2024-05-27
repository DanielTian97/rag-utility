import pandas as pd

class query_document_pair:
    def __init__(self, qid: str, docno: str, qText: str, dText: str, qrel: int, score: float):
        self.qid = qid
        self.docno = docno
        self.qText = qText
        self.dText = dText
        self.qrel = qrel
        self.rsv = score
        self.pred = None

    def put_prediction(self, pred: float):
        self.pred = pred
    
    def __str__(self) -> str:
        return f'qid_{self.qid}: {self.qText}; docno_{self.docno}; qrel: {self.qrel}; bm25_rsv: {self.rsv}.'

def get_msmarco_passage_pairs():
    get_msmarco_passage_pairs('msmarco_passage')

def get_msmarco_passage_pairs(dataset: str):
    dl_19_res_df = pd.read_csv('./res/bm25_dl_19.csv')
    dl_20_res_df = pd.read_csv('./res/bm25_dl_20.csv')

    dl_19_qids = dl_19_res_df.qid.unique()
    dl_20_qids = dl_20_res_df.qid.unique()

    with open(f'./middle_products/{dataset}.pkl', 'rb') as f:
        import pickle
        msmarco_doc_dict = pickle.load(f)
        f.close()
        
    queries = pd.read_csv('./middle_products/queries.csv')
    qrels = pd.read_csv('./middle_products/qrels.csv')
    qrels['docno'] = qrels.docno.astype('str')

    q_d_pair_list = []
    for qid in dl_19_qids:
        
        qText = queries[queries.qid==qid]['query'].tolist()[0]
        df_for_qid = dl_19_res_df[dl_19_res_df.qid == qid].sort_values(['rank'], ascending=True)
        denoted_docnos = qrels[qrels.qid == qid].docno.tolist()
        
        for docno, score in df_for_qid[['docno', 'score']].values[:50]:
            docno=str(int(docno))
            dText = msmarco_doc_dict[docno]
            
            if(docno in denoted_docnos):
                qrel = qrels[(qrels.qid==qid) & (qrels.docno==docno)].label.tolist()[0]
            else:
                qrel = -1
            
            q_d_pair = query_document_pair(qid=str(qid), docno=str(docno), qText=qText, dText=dText, qrel=qrel, score=float(score))
            q_d_pair_list.append(q_d_pair)
            # print(q_d_pair)
        
        del(df_for_qid)

    return q_d_pair_list
# print(q_d_pair_list)