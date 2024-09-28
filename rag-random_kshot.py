from llama_cpp import Llama
from compose_prompts import *
from llama_tools import llama_tools
import json
import sys
    
def compose_context(res, qid: str, batch_size, batch_step, top_starts, tail_starts, doc_dict):
      print(qid)
      retrieved_for_q = res[res.qid==qid]
      retrieved_num = retrieved_for_q['rank'].max()+1
      
      starts = list(range(0, (retrieved_num-1)-(batch_size-1)+1, batch_step))
      start_rank_list = list(set(starts[:top_starts]).union(set(starts[(len(starts)-1)-(tail_starts-1):])))
      start_rank_list.sort()
      print(start_rank_list)
      context_book = []
      for start in start_rank_list:
            context = ''
            end = start + batch_size
            batch_docnos = retrieved_for_q[(retrieved_for_q['rank']>=start)&(retrieved_for_q['rank']<end)].docno.tolist()
            batch_texts = [doc_dict[str(docno)] for docno in batch_docnos]
            
            num = 0
            for text in batch_texts:
                  num += 1
                  context += f'Context {num}: {text};\n'
            
            context_book.append(context)
            
      return start_rank_list, context_book

def update_json_result_file(file_name, result_to_write):
      f = open(file_name, "w+", encoding='UTF-8')
      json.dump(result_to_write, f, indent=4)
      f.close()

# prepare needed files
def prepare_data(dataset_name: str):
      # read the retrieved documents
      import pickle
      with open('./middle_products/msmarco_passage_v1_retrieved_top_tail.pkl', 'rb') as f:
            doc_dict = pickle.load(f)
            f.close()
      # prepare queries
      queries = pd.read_csv(f'./middle_products/queries_{dataset_name}.csv')
      # prepare res file
      res = pd.read_csv(f'./res/bm25_dl_{dataset_name}.csv') # retrieval result
      
      return doc_dict, queries, res

if __name__=="__main__":
      if(len(sys.argv) != 8):
            print("This experiment takes 7 parameters: ")
            print("1.batch size\n2.batch step\n3.num of calls\n4.top of starts\n5.tail of starts\n6.temperature\n7.19/20")
            print("e.g. 1 1 1 10 0 0.3 19")

      batch_size = int(sys.argv[1])
      batch_step = int(sys.argv[2])
      num_calls = int(sys.argv[3])
      # start control parameters
      top_starts = int(sys.argv[4])
      tail_starts = int(sys.argv[5])
      temperature = float(sys.argv[6])
      dataset_name = str(sys.argv[7])
      
      # load the llm
      llm = llama_tools.load_llama()
      # load needed data
      doc_dict, queries, res = prepare_data(dataset_name)
      
      setting_file_name = f'./middle_products/random_answers_{batch_size}shot_{num_calls}calls_dl_{dataset_name}_settings.json'
      setting_record = {'batch_size':batch_size, 'batch_step':batch_step, 'num_calls':num_calls, \
                  'top_starts':top_starts, 'tail_starts':tail_starts, 'temperature':temperature}
      f = open(setting_file_name, "w+", encoding='UTF-8')
      json.dump(setting_record, f, indent=4)
      f.close()

      file_name = f'./middle_products/random_answers_{batch_size}shot_{num_calls}calls_dl_{dataset_name}.json'
      # result_to_write = {} #{qid:result_for_qid}

      try:
            f = open(file=file_name, mode="r")
            result_to_write = json.load(f)
            existed_qids_list = list(result_to_write.keys())
            print(existed_qids_list)
            existed_qids = len(result_to_write)
            f.close()
      except:
            f = open(file=file_name, mode="w+")
            result_to_write= {}
            existed_qids_list = []
            existed_qids = 0
            f.close()

      preamble = "Please answer this question based on the given context. End your answer with STOP."

      q_no = 0
      for qid, query in zip(queries['qid'].tolist(), queries['query'].tolist()):
            print(f'q_number={q_no}--{qid}')
            q_no += 1
            if(str(qid) not in existed_qids_list):
                  varying_context_result = {} #{start: results}
                  existing_starts = []
            else:
                  varying_context_result = result_to_write[str(qid)] #added 0824
                  existing_starts = list(varying_context_result.keys()) #added 0824
                  print(existing_starts)

            start_records, context_book = compose_context(qid=qid, res=res, batch_size=batch_size, batch_step=batch_step, top_starts=top_starts, tail_starts=tail_starts, doc_dict=doc_dict)
            for start, context in zip(start_records, context_book):
                  llm.set_seed(1000) # added 0824
                  if(str(start) in existing_starts):
                        continue
                  print(f'\tstart_rank.{start}')
                  prompt = f'{preamble} \n{context}Question: \'{query}\' \nAnswer: '
                  print(prompt)
                  multi_call_results = {}
                  for j in range(num_calls):
                        print(f'\t\tno.{j}')
                        result = llama_tools.single_call(llm=llm, prompt=prompt, temperature=temperature)
                        multi_call_results.update({j: result})
                  varying_context_result.update({start: multi_call_results})
                        
            result_to_write.update({qid: varying_context_result})              
            update_json_result_file(file_name=file_name, result_to_write=result_to_write)