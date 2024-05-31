import pytrec_eval
from scipy import stats

def per_query_analysis(qrel_book, pred_book, predq_book, rsv_book):
    # print(qrel_book)
    print("DOWNSTREAM ANALYSES")

    name_mapping = {'ndcg_cut_10': 'nDCG@10', 'ndcg_cut_100': 'nDCG@100', 
                    'map_cut_10': 'MAP"10', 'map_cut_100': 'MAP@100'}
    qrel_evaluator = pytrec_eval.RelevanceEvaluator(qrel_book, {'ndcg_cut_10', 'ndcg_cut_100', 'map_cut_10', 'map_cut_100'})
    pseudo_evaluator = pytrec_eval.RelevanceEvaluator(predq_book, {'ndcg_cut_10', 'ndcg_cut_100', 'map_cut_10', 'map_cut_100'})
    ground_truth = qrel_evaluator.evaluate(rsv_book)
    pseudo_truth = pseudo_evaluator.evaluate(rsv_book)
    
    for metric in name_mapping:
    
        gt_list = []
        ps_list = []
    
        for qid in ground_truth:
            gt = ground_truth[qid]
            ps = pseudo_truth[qid]
        
            gt_list.append(gt[metric])
            ps_list.append(ps[metric])
    
            print(f'{name_mapping[metric]}:\t{stats.kendalltau(gt_list, ps_list)}')
    