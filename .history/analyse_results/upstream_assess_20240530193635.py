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
    qrel_evaluator = pytrec_eval.RelevanceEvaluator(qrel_book, {'ndcg_cut_10', 'ndcg_cut_20', 'ndcg_cut_50', 'ndcg_cut_80', 'ndcg_cut_100'})
    
    pred_books = {10: {}, 20: {}, 50: {}, 80: {}, 100: {}}
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
                
    print(f"Rerank cutoff\tnDCG@10\tnDCG@20\tnDCG@50\tnDCG@80\tnDCG@100")
    for c, pred_book in pred_books.items():
        
        evaluation_result = qrel_evaluator.evaluate(pred_book)
        ndcg_10_list = []
        ndcg_20_list = []
        ndcg_50_list = []
        ndcg_80_list = []
        ndcg_100_list = []
        
        for record in evaluation_result.values():
            ndcg_10_list.append(record['ndcg_cut_10'])
            ndcg_20_list.append(record['ndcg_cut_20'])
            ndcg_50_list.append(record['ndcg_cut_50'])
            ndcg_80_list.append(record['ndcg_cut_80'])
            ndcg_100_list.append(record['ndcg_cut_100'])
        
        avg_ndcg_10 = sum(ndcg_10_list)/len(ndcg_10_list)
        avg_ndcg_20 = sum(ndcg_20_list)/len(ndcg_20_list)
        avg_ndcg_50 = sum(ndcg_50_list)/len(ndcg_50_list)
        avg_ndcg_80 = sum(ndcg_80_list)/len(ndcg_80_list)
        avg_ndcg_100 = sum(ndcg_100_list)/len(ndcg_100_list)
        
        output_string = f"{c}"
        if(c >= 10):
            output_string += f"\t{avg_ndcg_10}"
            if(c >= 20):
                output_string += f"\t{avg_ndcg_20}"
                if(c >= 50):
                    output_string += f"\t{avg_ndcg_50}"
                    if(c >= 80):
                        output_string += f"\t{avg_ndcg_80}"
                        if(c >= 100):
                            output_string += f"\t{avg_ndcg_100}"
        print(output_string)
