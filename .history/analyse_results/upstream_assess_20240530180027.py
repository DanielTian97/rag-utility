import math
from scipy import stats
import pytrec_eval

def ndcg_pred_qrel(result: list, qrel_book):
    print("UPSTREAM ANALYSES")
    print("----Statistics about the prediction for different groups of query-document pairs:")
    list_score = {-1: [], 0: [], 1: [], 2: [], 3: []}
    qid_pair_dict = {}
    for record in result:
        list_score[record.qrel].append(record.pred)
        if(record.qid not in qid_pair_dict.keys()):
            qid_pair_dict.update({str(record.qid): [record]})
        else:
            qid_pair_dict[record.qid].append(record)

    print(f'qrel_group\tmean(pred)\tmin(pred)\tmax(pred)')
    for i in list_score:
        print(f'{i}\t{sum(list_score[i])/len(list_score[i])}\t{min(list_score[i])}\t{max(list_score[i])}')
    
    print("----Per-query correlation between LLM score and ground truth")
    qrel_evaluator = pytrec_eval.RelevanceEvaluator(qrel_book, {'ndcg_cut_10', 'ndcg_cut_100'})
    
    pred_books = {10: {}, 20: {}, 50: {}, 100: {}}
    for record in result:
        qid = record.qid
        docno = record.docno
        pred = float(record.pred)
        
        if(qid not in pred_books[10]):
            for c in pred_books.keys():
                pred_books[c].update({qid: {}})
        
        for c in pred_books.keys():
            if(len(pred_books[c][qid]) < c):
                pred_books[c][qid].update({docno: pred})
    
    for c, pred_book in pred_books.items():
        print(f"Rerank cutoff = {c}")
        print(qrel_evaluator.evaluate(pred_book))
