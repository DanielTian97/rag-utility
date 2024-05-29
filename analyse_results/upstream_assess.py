import math
from scipy import stats

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
            qrel_list.append(max(0, i))

    import tensorflow_ranking as tfr

    # print("Calculating ndcg between LLM score and real score ....")
    # true = [qrel_list]
    # pred = [pred_list]
    ndcg_10 = tfr.keras.metrics.NDCGMetric(topn=10)
    ndcg_50 = tfr.keras.metrics.NDCGMetric(topn=50)
    ndcg_100 = tfr.keras.metrics.NDCGMetric(topn=100)
    ndcg_1000 = tfr.keras.metrics.NDCGMetric(topn=1000)
    ndcg = tfr.keras.metrics.NDCGMetric(topn=len(qrel_list))
    # print(f'10\t{ndcg_10(true, pred).numpy()}')
    # print(f'100\t{ndcg_100(true, pred).numpy()}')
    # print(f'1000\t{ndcg_1000(true, pred).numpy()}')
    # print(f'full\t{ndcg(true, pred).numpy()}')
    
    print("Calculating AVERAGE ndcg between LLM score and real score across individual queries....")
    print(f'\ttau\t10\t50\t100')
    ndcg_10_per_query = []
    ndcg_50_per_query = []
    ndcg_100_per_query = []
    tau_value_per_query = []
    
    for qid, pairs in qid_pair_dict.items():
        # print(qid, len(pairs))
        true = []
        pred = []
        for pair in pairs[:rerank_cutoff]:
            true.append(max(pair.qrel, 0))
            pred.append(pair.pred)
        
        ndcg_10_value = ndcg_10([true], [pred]).numpy()
        tau_value = float(stats.kendalltau(true, pred)[0])
        
        output_string = f'{qid}\t{tau_value}\t{ndcg_10_value}'
        
        ndcg_10_per_query.append(ndcg_10_value)
        if(not math.isnan(tau_value)):
            tau_value_per_query.append(tau_value)
        
        if(rerank_cutoff >= 50):
            ndcg_50_value = ndcg_50([true], [pred]).numpy()
            output_string += f'\t{ndcg_50_value}'
            ndcg_50_per_query.append(ndcg_50_value)
            
            if(rerank_cutoff >= 100):
                ndcg_100_value = ndcg_100([true], [pred]).numpy()
                output_string += f'\t{ndcg_100_value}'
                ndcg_100_per_query.append(ndcg_100_value)
        
        print(output_string)
        
    if(len(ndcg_10_per_query) > 0):
        output_string = f'avg\t{sum(tau_value_per_query)/len(tau_value_per_query)}\t{sum(ndcg_10_per_query)/len(ndcg_10_per_query)}'
        if(len(ndcg_50_per_query) > 0):
            output_string += f'\t{sum(ndcg_50_per_query)/len(ndcg_50_per_query)}'
            if(len(ndcg_100_per_query) > 0):
                output_string += f'\t{sum(ndcg_100_per_query)/len(ndcg_100_per_query)}'
        print(output_string)