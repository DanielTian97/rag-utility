import tensorflow_ranking as tfr
from scipy import stats

def correlation(result: list, evaluaton_gt, evaluation_pred, query_sets: list):
    qid_pair_dict = {}
    for record in result:
        if(record.qid not in qid_pair_dict.keys()):
            qid_pair_dict.update({str(record.qid): [record]})
        else:
            qid_pair_dict[record.qid].append(record)
            
    ndcg_10 = tfr.keras.metrics.NDCGMetric(topn=10)
    ndcg_100 = tfr.keras.metrics.NDCGMetric(topn=100)
    
    pred_10_list = []
    pred_100_list = []
    pred_ap_100_list = []
    
    t_ndcg_10_list = []
    t_ndcg_100_list = []
    
    gt_ndcg_10_list = []
    gt_ap_100_list = []
    ps_ndcg_10_list = []
    ps_ap_100_list = []
    
    gt_ndcg_10_list_0 = [] # 19 or 21
    gt_ndcg_10_list_1 = [] # 20 or 22
    ps_ndcg_10_list_0 = [] # 19 or 21
    ps_ndcg_10_list_1 = [] # 20 or 22

    for qid, pairs in qid_pair_dict.items():
        rsvs = []
        preds = []
        qrels = []
        for pair in pairs:
            rsvs.append(pair.rsv)
            preds.append(pair.pred)
            qrel = max(pair.qrel, 0)
            # if(qrel>=2):
            #     qrel = qrel
            # else:
            #     qrel = 0
            qrels.append(qrel)
                
        pred_10 = ndcg_10([preds], [rsvs]).numpy() #~=ndcg_10([true], [pred]).numpy()
        pred_100 = ndcg_100([preds], [rsvs]).numpy()
        pred_ap_100 = sum(preds)/len(preds)
        
        gt_ndcg_10 = evaluaton_gt[(evaluaton_gt.qid==qid)&(evaluaton_gt.measure=='nDCG@10')].value.values[0]
        gt_ap_100 = evaluaton_gt[(evaluaton_gt.qid==qid)&(evaluaton_gt.measure=='AP(rel=2)@100')].value.values[0]
        ps_ndcg_10 = evaluation_pred[(evaluation_pred.qid==qid)&(evaluation_pred.measure=='nDCG@10')].value.values[0]
        ps_ap_100 = evaluation_pred[(evaluation_pred.qid==qid)&(evaluation_pred.measure=='AP@100')].value.values[0]
        
        t_ndcg_10 = ndcg_10([qrels], [rsvs]).numpy()
        t_ndcg_100 = ndcg_100([qrels], [rsvs]).numpy()
        # print(qid, pred_10, pred_100, pred_ap_100, gt_ndcg_10, gt_ap_100)
        
        pred_10_list.append(pred_10)
        pred_100_list.append(pred_100)
        gt_ndcg_10_list.append(gt_ndcg_10)
        gt_ap_100_list.append(gt_ap_100)
        ps_ndcg_10_list.append(ps_ndcg_10)
        ps_ap_100_list.append(ps_ap_100)
        pred_ap_100_list.append(pred_ap_100)
        t_ndcg_10_list.append(t_ndcg_10)
        t_ndcg_100_list.append(t_ndcg_100)
        
        if(qid in query_sets[0]):
            gt_ndcg_10_list_0.append(gt_ndcg_10) # 19 or 21
            ps_ndcg_10_list_0.append(ps_ndcg_10) 
        elif(qid in query_sets[1]):
            gt_ndcg_10_list_1.append(gt_ndcg_10) 
            ps_ndcg_10_list_1.append(ps_ndcg_10) # 20 or 22
        
    # print(stats.kendalltau(pred_10_list, t_ndcg_10_list))
    # print(stats.kendalltau(pred_100_list, t_ndcg_100_list))
    # print(stats.kendalltau(gt_ndcg_10_list, t_ndcg_10_list))
    # print(stats.kendalltau(gt_ap_100_list, t_ndcg_100_list))
    print(f'full\t{stats.kendalltau(gt_ndcg_10_list, ps_ndcg_10_list)}')
    # print(stats.kendalltau(gt_ap_100_list, ps_ap_100_list))
    print(f'set(0)\t{stats.kendalltau(gt_ndcg_10_list_0, ps_ndcg_10_list_0)}')
    print(f'set(1)\t{stats.kendalltau(gt_ndcg_10_list_1, ps_ndcg_10_list_1)}')