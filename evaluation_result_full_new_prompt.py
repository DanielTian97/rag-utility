from evaluate import load
import pandas as pd
import pickle
import json
import sys

def prepare_qids_qrels_docdict(dataset_name):

    with open('./middle_products/msmarco_passage_v1_qrels.pkl', 'rb') as f:
        doc_dict = pickle.load(f)
    
    queries = pd.read_csv(f'./middle_products/queries_{dataset_name}.csv')
    queries['qid'] = queries['qid'].astype('str')
    qids = queries.qid.tolist()
    qrels = pd.read_csv('./middle_products/qrels.csv')
    qrels['qid'] = qrels['qid'].astype('str')
    qrels['docno'] = qrels['docno'].astype('str')
    
    return qids, qrels, doc_dict

def get_docnos(qid, doc_dict, qrels):
    docnos_0 = [doc_dict[str(docno)] for docno in qrels[(qrels.qid==qid)&(qrels.label==0)].docno.tolist()]
    docnos_1 = [doc_dict[str(docno)] for docno in qrels[(qrels.qid==qid)&(qrels.label==1)].docno.tolist()]
    docnos_2 = [doc_dict[str(docno)] for docno in qrels[(qrels.qid==qid)&(qrels.label==2)].docno.tolist()]
    docnos_3 = [doc_dict[str(docno)] for docno in qrels[(qrels.qid==qid)&(qrels.label==3)].docno.tolist()]

    docno_dict = {0: docnos_0, 1: docnos_1, 2: docnos_2, 3: docnos_3}
    return docno_dict

def evaluator(to_eval: str, docno_dict: dict, qrel_level: int):
    print(f'\t\t\t{qrel_level}')
    
    doc_texts = docno_dict[qrel_level]
    if(len(doc_texts)==0):
        r = {\
            'precision': {'avg': -1, 'max': -1},\
            'recall': {'avg': -1, 'max': -1},\
            'f1': {'avg': -1, 'max': -1},\
            }
        return r

    pred_text = to_eval
    predictions = len(doc_texts)*[pred_text]
    references = doc_texts
    results = bertscore.compute(predictions=predictions, references=references, lang="en", model_type="bert-large-uncased", verbose=False)

    precisions, recall, f1 = results['precision'], results['recall'], results['f1']

    # print('precision', sum(precisions)/len(precisions), max(precisions))
    # print('recall', sum(recall)/len(recall), max(recall))
    # print('f1', sum(f1)/len(f1), max(f1))
    r = {\
        'precision': {'avg': sum(precisions)/len(precisions), 'max': max(precisions)},\
        'recall': {'avg': sum(recall)/len(recall), 'max': max(recall)},\
        'f1': {'avg': sum(f1)/len(f1), 'max': max(f1)},\
        }
    return r
    
def eval_by_qrels(to_eval: str, docno_dict):
    # print(to_eval)
    r2 = evaluator(to_eval, docno_dict, 2)
    r3 = evaluator(to_eval, docno_dict, 3)
    r = {'qrel_2': r2, 'qrel_3': r3}
    return r

if __name__=="__main__":
    
    batch_size = int(sys.argv[1])
    num_calls = sys.argv[2]
    dataset_name = sys.argv[3]
    top_num = sys.argv[4]
    tail_num = sys.argv[5]
    
    suffix = '_prompt1'
    if(len(sys.argv) >= 7):
        suffix = sys.argv[6]
        
    retriever_name = 'bm25'
    file_path = f'./middle_products/random_answers_{batch_size}shot_{num_calls}calls_{top_num}_{tail_num}_dl_{dataset_name}{suffix}.json'
    eval_file_path = f'./eval_results/random_answers_{batch_size}shot_{num_calls}calls_{top_num}_{tail_num}_dl_{dataset_name}{suffix}_eval.json'
    
    if(len(sys.argv) >= 8):
        retriever_name = sys.argv[7]
        file_path = f'./middle_products/random_answers_{batch_size}shot_{num_calls}calls_{top_num}_{tail_num}_{retriever_name}_dl_{dataset_name}{suffix}.json'
        eval_file_path = f'./eval_results/random_answers_{batch_size}shot_{num_calls}calls_{top_num}_{tail_num}_{retriever_name}_dl_{dataset_name}{suffix}_eval.json'

    # experiment begins
    bertscore = load("bertscore")
    # prepare data
    qids, qrels, doc_dict = prepare_qids_qrels_docdict(dataset_name)
    # read the generated answers
    # batch_size = 1
    # num_calls = 5

    with open(file=file_path, mode="r") as f:
        answer_book = json.load(f)
        f.close()
    
    # create the file
    try:
        f = open(file=eval_file_path, mode="r")
        existed_results = json.load(f)
        existed_qids = len(existed_results)
        f.close()
    except:
        f = open(file=eval_file_path, mode="w+")
        existed_results = {}
        existed_qids = 0
        f.close()
        
    # eval_result_dict = {}
    for qid in [str(id) for id in qids[existed_qids:]]:
        print(f'Qid={qid}')
        eval_result_qid = {}
        for start in answer_book[str(qid)].keys():
            print(f'\tstart={start}, batch={batch_size}')
            eval_result_start = {}
            for i in answer_book[str(qid)][str(start)].keys():
                print(f'\t\t{i}')
                to_eval = answer_book[str(qid)][str(start)][str(i)]['answer']
                docno_dict = get_docnos(qid=qid, doc_dict=doc_dict, qrels=qrels)

                r = eval_by_qrels(to_eval=to_eval, docno_dict=docno_dict)

                eval_result_start.update({i: r})
            eval_result_qid.update({start: eval_result_start})
        
        # eval_result_dict.update({qid: eval_result_qid})

        with open(file=eval_file_path, mode="r") as f:
            # existed_results = json.load(f)
            existed_results.update({qid: eval_result_qid})
            f.close()

        with open(file=eval_file_path, mode="w+") as f:
            json.dump(existed_results, f, indent=4)
            f.close()
        
            