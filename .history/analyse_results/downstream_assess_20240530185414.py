import pytrec_eval
from scipy import stats

def per_query_analysis(qrel_book, pred_book, predq_book, rsv_book):
    # print(qrel_book)
    print("DOWNSTREAM ANALYSES")

    qrel_evaluator = pytrec_eval.RelevanceEvaluator(qrel_book, {'ndcg_cut_10', 'ndcg_cut_100', 'map_cut_10', 'map_cut_100'})
    pseudo_evaluator = pytrec_eval.RelevanceEvaluator(predq_book, {'ndcg_cut_10', 'ndcg_cut_100', 'map_cut_10', 'map_cut_100'})
    ground_truth = qrel_evaluator.evaluate(rsv_book)
    pseudo_truth = pseudo_evaluator.evaluate(rsv_book)
    
    gt_ndcg10_list = []
    gt_ndcg100_list = []
    ps_ndcg10_list = []
    ps_ndcg100_list = []
    
    gt_map10_list = []
    gt_map100_list = []
    ps_map10_list = []
    ps_map100_list = []
    
    for qid in ground_truth:
        gt = ground_truth[qid]
        ps = pseudo_truth[qid]
        
        gt_ndcg10_list.append(gt['ndcg_cut_10'])
        gt_ndcg100_list.append(gt['ndcg_cut_100'])
        ps_ndcg10_list.append(ps['ndcg_cut_10'])
        ps_ndcg100_list.append(ps['ndcg_cut_100'])
        
        gt_map10_list.append(gt['map_cut_10'])
        gt_map100_list.append(gt['map_cut_100'])
        ps_map10_list.append(ps['map_cut_10'])
        ps_map100_list.append(ps['map_cut_100'])
    
    print(f'nDCG@10:\t{stats.kendalltau(gt_ndcg10_list, ps_ndcg10_list)}')
    print(f'nDCG@100:\t{stats.kendalltau(gt_ndcg100_list, ps_ndcg100_list)}')
    print(f'MAP@10:\t{stats.kendalltau(gt_map10_list, ps_map10_list)}')
    print(f'MAP@100:\t{stats.kendalltau(gt_map100_list, ps_map100_list)}')