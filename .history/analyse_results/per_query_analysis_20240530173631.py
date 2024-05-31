import pytrec_eval

def per_query_analysis(qrel_book, pred_book, predq_book, rsv_book):
    print(qrel_book)

    qrel_evaluator = pytrec_eval.RelevanceEvaluator(qrel_book, {'ndcg_cut_10', 'ndcg_cut_100'})
    pseudo_evaluator = pytrec_eval.RelevanceEvaluator(predq_book, {'ndcg_cut_10', 'ndcg_cut_100'})

    print(qrel_evaluator.evaluate(rsv_book))
    print(qrel_evaluator.evaluate(pred_book))
    print(pseudo_evaluator.evaluate(rsv_book))