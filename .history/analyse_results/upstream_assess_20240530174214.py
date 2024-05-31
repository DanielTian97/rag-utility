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

    print(f'qrel_group\tmean(pred)\tmin(pred)\tmax(pred)')
    for i in list_score:
        print(f'{i}\t{sum(list_score[i])/len(list_score[i])}\t{min(list_score[i])}\t{max(list_score[i])}')

