from tools.permutation_generator import *

def used_preamble(long_answer=True): # for k-shot
    if(long_answer):
        return "You are an expert at answering questions based on your own knowledge and related context. Please answer this question based on the given context. End your answer with STOP."
    else:
        return "You are given a question and you MUST respond by generating the answer (max 5 tokens) based on your own knowledge and related context. End your answer with STOP."
        
def used_preamble_0(long_answer=True): # for 0-shot
    if(long_answer):
        return "You are an expert at answering questions based on your own knowledge. Please answer this question. End your answer with STOP."
    else:
        return "You are given a question and you MUST respond by generating the answer (max 5 tokens) based on your own knowledge. End your answer with STOP."

# prepare needed files
def prepare_data(dataset_name: str, retriever_name = 'bm25'):
    import pandas as pd
    # read the retrieved documents
    import pickle

    if((dataset_name=='19')|(dataset_name=='20')|(dataset_name=='dev_small')):
        with open('./doc_dicts/msmarco_passage_dict.pkl', 'rb') as f:
            doc_dict = pickle.load(f)
            f.close()
    elif((dataset_name=='21')|(dataset_name=='22')):
        with open('./doc_dicts/msmarco_passage_v2_dict.pkl', 'rb') as f:
            doc_dict = pickle.load(f)
            f.close()
    elif(dataset_name=='nq_test'):
        with open('./doc_dicts/nq_wiki_dict.pkl', 'rb') as f:
            doc_dict = pickle.load(f)
            f.close()       
    else:
        print('this dataset is not supported')
        return
    
    # prepare queries
    queries = pd.read_csv(f'./queries/queries_{dataset_name}.csv')
    # prepare res file

    if((dataset_name=='dev_small')|(dataset_name=='nq_test')):
        res = pd.read_csv(f'./res/{retriever_name}_{dataset_name}.csv') # retrieval result
    else:
        res = pd.read_csv(f'./res/{retriever_name}_dl_{dataset_name}.csv') # retrieval result
      
    return doc_dict, queries, res

# compose the examples in the context part
def compose_context(res, qid: str, k, step, tops, tails, doc_dict, reverse_order=False):
    print(qid)
    res.qid = res.qid.astype('str')
    retrieved_for_q = res[res.qid==str(qid)]

    retrieved_num = retrieved_for_q['rank'].max()+1

    try:
        starts = list(range(0, (retrieved_num-1)-(k-1)+1, step))
    except:
        starts = []
        
    start_rank_list = list(set(starts[:tops]).union(set(starts[(len(starts)-1)-(tails-1):])))
    start_rank_list.sort()
    print(start_rank_list)
    context_book = []
    for start in start_rank_list:
        context = ''
        end = start + k
        batch_docnos = retrieved_for_q[(retrieved_for_q['rank']>=start)&(retrieved_for_q['rank']<end)].docno.tolist()
        batch_texts = [doc_dict[str(docno)] for docno in batch_docnos]
        if(reverse_order):
            batch_texts = list(reversed(batch_texts))
            
        num = 0
        for text in batch_texts:
            num += 1
            context += f'Context {num}: "{text}";\n'
            
        context_book.append(context)
            
    return start_rank_list, context_book

def compose_context_with_permutations(res, qid: str, k, step, tops, tails, doc_dict, full_permutations):
    
    print(qid)
    retrieved_for_q = res[res.qid==qid]
    retrieved_num = retrieved_for_q['rank'].max()+1
      
    starts = list(range(0, (retrieved_num-1)-(k-1)+1, step))
    start_rank_list = list(set(starts[:tops]).union(set(starts[(len(starts)-1)-(tails-1):])))
    print(start_rank_list)
    start_rank_list.sort()
      
    p_name_list = []
    context_book = []
    for start in start_rank_list:
        end = start + k
        batch_docnos = retrieved_for_q[(retrieved_for_q['rank']>=start)&(retrieved_for_q['rank']<end)].docno.tolist()

        permuntation_docnos = get_permutation(batch_docnos, len(batch_docnos), full_permutations=full_permutations)
            
        for p_name, p_batch_docnos in permuntation_docnos.items():
            context = ''
                  
            batch_texts = [doc_dict[str(docno)] for docno in p_batch_docnos]
            num = 0
            for text in batch_texts:
                num += 1
                context += f'Context {num}: "{text}";\n'
                  
            p_name_list.append(f'{start}>{p_name}')
            context_book.append(context)
            
    return p_name_list, context_book

def prompt_assembler_0(query:str, long_answer=True):
    preamble = used_preamble_0(long_answer)
    return f'{preamble} \nQuestion: "{query}"\nNow start your answer. \nAnswer: '

def prompt_assembler(context:str, query:str, long_answer=True):
    preamble = used_preamble(long_answer)
    return f'{preamble} \n{context}Question: "{query}"\nNow start your answer. \nAnswer: '