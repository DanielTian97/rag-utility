from llama_cpp import Llama
from compose_prompts import *
from llama_tools import llama_tools
from experiment_tools import *
from prompt1_tools import *
import json
import sys
from permutation_generator import *

if __name__=="__main__":
      if(len(sys.argv) != 9):
            print("This experiment takes 8 parameters: ")
            print("1.batch size\n2.batch step\n3.num of calls\n4.top of starts\n5.tail of starts\ntemperature\nwhether_exam_full_permutations\n19/20")
            print("e.g. 1 1 1 10 0 0.2 False")

      batch_size = int(sys.argv[1])
      batch_step = int(sys.argv[2])
      num_calls = int(sys.argv[3])
      # start control parameters
      top_starts = int(sys.argv[4])
      tail_starts = int(sys.argv[5])
      temperature = float(sys.argv[6])
      full_permutation = eval(sys.argv[7])
      print('input', full_permutation)
      dataset_name = str(sys.argv[8])
      
      # load the llm
      llm = llama_tools.load_llama()
      # load needed data
      doc_dict, queries, res = prepare_data(dataset_name)
      
      setting_file_name = f'./middle_products/random_answers_{batch_size}shot_{num_calls}calls_{top_starts}_{tail_starts}_dl_{dataset_name}_settings_p_prompt1.json'
      setting_record = {'batch_size':batch_size, 'batch_step':batch_step, 'num_calls':num_calls, \
                  'top_starts':top_starts, 'tail_starts':tail_starts, 'temperature':temperature}
      f = open(setting_file_name, "w+", encoding='UTF-8')
      json.dump(setting_record, f, indent=4)
      f.close()

      file_name = f'./middle_products/random_answers_{batch_size}shot_{num_calls}calls_{top_starts}_{tail_starts}_dl_{dataset_name}_p_prompt1.json'
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

      preamble = used_preamble()
      q_no = 0
      for qid, query in zip(queries['qid'].tolist(), queries['query'].tolist()):
            print(f'q_number={q_no}--{qid}')
            if(str(qid) in existed_qids_list):
                  print("Already generated, next!")
                  continue
            q_no += 1
            varying_context_result = {} #{start: results}
            
            start_records, context_book = compose_context_with_permutations(qid=qid, res=res, batch_size=batch_size, batch_step=batch_step, \
                  top_starts=top_starts, tail_starts=tail_starts, doc_dict=doc_dict, full_permutations=full_permutation)
            print('start records: ', start_records)

            for start, context in zip(start_records, context_book):
                  print(f'\tstart_rank.{start}')
                  prompt = f'{preamble} \n{context}Question: "{query}"\nNow start your answer. \nAnswer: '
                  print(prompt)
                  multi_call_results = {}
                  varying_context_result.update({start: multi_call_results})
                  
                  for j in range(num_calls):
                        print(f'\t\tno.{j}')
                        result = llama_tools.single_call(llm=llm, prompt=prompt, temperature=temperature)
                        multi_call_results.update({j: result})
                        
            result_to_write.update({qid: varying_context_result})              
            update_json_result_file(file_name=file_name, result_to_write=result_to_write)
            
      del llm