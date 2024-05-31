import pytrec_eval

def per_query_analysis(qrel_book, pred_book, predq_book, rsv_book):
    # print(qrel_book)
    print("PER-QUERY ANALYSES")

    qrel_evaluator = pytrec_eval.RelevanceEvaluator(qrel_book, {'ndcg_cut_10', 'ndcg_cut_100'})
    pseudo_evaluator = pytrec_eval.RelevanceEvaluator(predq_book, {'ndcg_cut_10', 'ndcg_cut_100'})

    ground_truth = qrel_evaluator.evaluate(rsv_book)
    pseudo_truth = pseudo_evaluator.evaluate(rsv_book)
    
    gt_10_list = []
    gt_100_list = []
    ps_10_list = []
    ps_100_list = []
    
    for qid in ground_truth:
        gt = ground_truth[qid]
        ps = pseudo_truth[qid]
        
        