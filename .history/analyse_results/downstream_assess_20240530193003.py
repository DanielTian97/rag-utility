import pytrec_eval
from scipy import stats

def per_query_analysis(qrel_book, predq_book, rsv_book, queries_0, queries_1):
    # print(qrel_book)
    print("DOWNSTREAM ANALYSES")

    name_mapping = {'ndcg_cut_10': 'nDCG@10', 'ndcg_cut_100': 'nDCG@100', 
                    'map_cut_10': 'MAP@10', 'map_cut_100': 'MAP@100'}
    qrel_evaluator = pytrec_eval.RelevanceEvaluator(qrel_book, {'ndcg_cut_10', 'ndcg_cut_100', 'map_cut_10', 'map_cut_100'})
    pseudo_evaluator = pytrec_eval.RelevanceEvaluator(predq_book, {'ndcg_cut_10', 'ndcg_cut_100', 'map_cut_10', 'map_cut_100'})
    ground_truth = qrel_evaluator.evaluate(rsv_book)
    pseudo_truth = pseudo_evaluator.evaluate(rsv_book)
    
    print(f'Metric:\tQuerySet_0\tQuerySet_1\tFull')
    for metric in name_mapping:
    
        gt_list = []
        ps_list = []
        
        gt_list_0 = []
        ps_list_0 = []
        
        gt_list_1 = []
        ps_list_1 = []
    
        for qid in ground_truth:
            gt = ground_truth[qid]
            ps = pseudo_truth[qid]
        
            gt_list.append(gt[metric])
            ps_list.append(ps[metric])
            
            if(str(qid) in queries_0):
                gt_list_0.append(gt[metric])
                ps_list_0.append(ps[metric])
            elif(str(qid) in queries_1):
                gt_list_1.append(gt[metric])
                ps_list_1.append(ps[metric])
                
        print(f'{name_mapping[metric]}:',
              f'\t{stats.kendalltau(gt_list_0, ps_list_0)[0]}',
              f'\t{stats.kendalltau(gt_list_1, ps_list_1)[0]}',
              f'\t{stats.kendalltau(gt_list, ps_list)[0]}')
    