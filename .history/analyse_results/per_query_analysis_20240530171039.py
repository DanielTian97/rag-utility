import pytrec_eval

def per_query_analysis(qrel_book, pred_book, predq_book, rsv_book):

    qrel_evaluator = pytrec_eval.RelevanceEvaluator(qrel_book, {'ndcg_cut_10'}, {'ndcg_cut_100'})
    pseudo_evaluator = pytrec_eval.RelevanceEvaluator(predq_book, {'ndcg_cut_10'}, {'ndcg_cut_100'})

    print(evaluator.evaluate(run)['q1'])