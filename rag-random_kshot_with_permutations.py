from tools import llama_tools, prompt_tools, experiment_tools
import json
import sys
from permutation_generator import *

if __name__=="__main__":
      if(len(sys.argv) < 10):
            print("This experiment takes 9 parameters: ")
            print("1.number of context passages\n2.step\n3.num of calls\n4.top of starts\n5.tail of starts\n6.temperature\n7.whether_exam_full_permutations\n8.19/20\n9.retriever name (if not specified it will be bm25)")
            print("e.g. 1 1 1 10 0 0.2 False 19")

      k = int(sys.argv[1])
      step = int(sys.argv[2])
      num_calls = int(sys.argv[3])
      # start control parameters
      top_starts = int(sys.argv[4])
      tail_starts = int(sys.argv[5])
      temperature = float(sys.argv[6])
      full_permutation = eval(sys.argv[7])
      print('input', full_permutation)
      dataset_name = str(sys.argv[8])
      
      retriever_name = 'bm25'
      if(len(sys.argv) == 10):
            retriever_name = str(sys.argv[9])
      
      # load the llm
      llm = llama_tools.load_llama()
      # load needed data
      doc_dict, queries, res = prompt_tools.prepare_data(dataset_name, retriever_name)
      
      setting_file_name = f'./gen_results/random_answers_{k}shot_{num_calls}calls_{top_starts}_{tail_starts}_{retriever_name}_dl_{dataset_name}_settings_p_prompt1.json'
      setting_record = {'k':k, 'step':step, 'num_calls':num_calls, \
                  'top_starts':top_starts, 'tail_starts':tail_starts, 'temperature':temperature}
      f = open(setting_file_name, "w+", encoding='UTF-8')
      json.dump(setting_record, f, indent=4)
      f.close()

      file_name = f'./gen_results/random_answers_{k}shot_{num_calls}calls_{top_starts}_{tail_starts}_{retriever_name}_dl_{dataset_name}_p_prompt1.json'
      # result_to_write = {} #{qid:result_for_qid}

      try:
            f = open(file=file_name, mode="r")
            result_to_write = json.load(f)
            existed_qids = len(result_to_write)
            existed_qids_list = list(result_to_write.keys())
            f.close()
      except:
            f = open(file=file_name, mode="w+")
            result_to_write= {}
            existed_qids = 0
            existed_qids_list = []
            f.close()

      q_no = 0
      for qid, query in zip(queries['qid'].tolist(), queries['query'].tolist()):
            print(f'q_number={q_no}--{qid}')
            if(str(qid) in existed_qids_list):
                  print("Already generated, next!")
                  continue
            q_no += 1
            varying_context_result = {} #{start: results}
            
            start_records, context_book = prompt_tools.compose_context_with_permutations(qid=qid, res=res, k=k, step=step, \
                  top_starts=top_starts, tail_starts=tail_starts, doc_dict=doc_dict, full_permutations=full_permutation)
            print('start records: ', start_records)

            for start, context in zip(start_records, context_book):
                  llm.set_seed(1000) # added 0824
                  print(f'\tstart_rank.{start}')
                  prompt = prompt_tools.prompt_assembler(context, query)
                  print(prompt)
                  multi_call_results = {}
                  varying_context_result.update({start: multi_call_results})
                  
                  for j in range(num_calls):
                        print(f'\t\tno.{j}')
                        result = llama_tools.single_call(llm=llm, prompt=prompt, temperature=temperature)
                        multi_call_results.update({j: result})
                        
            result_to_write.update({qid: varying_context_result})              
            experiment_tools.update_json_result_file(file_name=file_name, result_to_write=result_to_write)
            
      del llm