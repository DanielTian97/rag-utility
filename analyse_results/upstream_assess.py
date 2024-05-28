def ndcg_pred_qrel(result: list, rerank_cutoff: int):
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

    pred_list = []
    qrel_list = []
    print(f'qrel_group\tmean(pred)\tmin(pred)\tmax(pred)')
    for i in list_score:
        print(f'{i}\t{sum(list_score[i])/len(list_score[i])}\t{min(list_score[i])}\t{max(list_score[i])}')
        for v in list_score[i]:
            pred_list.append(v)
            qrel_list.append(max(0, min(i, 2)))

    import tensorflow_ranking as tfr

    print("Calculating ndcg between LLM score and real score ....")
    true = [qrel_list]
    pred = [pred_list]
    ndcg_10 = tfr.keras.metrics.NDCGMetric(topn=10)
    ndcg_50 = tfr.keras.metrics.NDCGMetric(topn=50)
    ndcg_100 = tfr.keras.metrics.NDCGMetric(topn=100)
    ndcg_1000 = tfr.keras.metrics.NDCGMetric(topn=1000)
    ndcg = tfr.keras.metrics.NDCGMetric(topn=len(qrel_list))
    print(f'10\t{ndcg_10(true, pred).numpy()}')
    print(f'100\t{ndcg_100(true, pred).numpy()}')
    print(f'1000\t{ndcg_1000(true, pred).numpy()}')
    print(f'full\t{ndcg(true, pred).numpy()}')
    
    print("Calculating AVERAGE ndcg between LLM score and real score across individual queries....")
    print(f'\t10\t50\t100')
    for qid, pairs in qid_pair_dict.items():
        # print(qid, len(pairs))
        true = []
        pred = []
        for pair in pairs[:rerank_cutoff]:
            true.append(pair.qrel)
            pred.append(pair.pred)
        
        output_string = f'{qid}\t{ndcg_10([true], [pred]).numpy()}'
        if(rerank_cutoff >= 50):
            output_string += f'\t{ndcg_50([true], [pred]).numpy()}'
            if(rerank_cutoff >= 100):
                output_string += f'\t{ndcg_100([true], [pred]).numpy()}'
        
        print(output_string)