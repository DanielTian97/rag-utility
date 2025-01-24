from tools import llama_tools, experiment_tools, prompt_tools
import json
import argparse

if __name__=="__main__":
      
      parser = argparse.ArgumentParser()
      parser.add_argument("--k", type=int, default=3)
      parser.add_argument("--step", type=int, default=1)
      parser.add_argument("--num_calls", type=int, default=5)
      parser.add_argument("--tops", type=int, default=1)
      parser.add_argument("--tails", type=int, default=0)
      parser.add_argument("--temperature", type=float, default=0.3)
      parser.add_argument("--dataset_name", type=str, choices=['19', '20', '21', '22'])
      parser.add_argument("--retriever", type=str, default='bm25', choices=['bm25', 'mt5', 'oracle', 'reverse_oracle'])
      args = parser.parse_args()

      signed_k = args.k
      step = args.step
      num_calls = args.num_calls
      # start control parameters
      top_starts = args.tops
      tail_starts = args.tails
      temperature = args.temperature
      dataset_name = args.dataset_name
      retriever_name = args.retriever
      
      k = signed_k
      reverse_order = k<0
      k = abs(k)
      
      # load the llm
      llm = llama_tools.load_llama()
      # load needed data
      doc_dict, queries, res = prompt_tools.prepare_data(dataset_name, retriever_name)
      queries.qid = queries.qid.astype('str')
      
      setting_file_name = f'./gen_results/random_answers_{signed_k}shot_{num_calls}calls_{top_starts}_{tail_starts}_{retriever_name}_dl_{dataset_name}_prompt1_settings.json'
      setting_record = {'k':signed_k, 'step':step, 'num_calls':num_calls, \
                  'top_starts':top_starts, 'tail_starts':tail_starts, 'temperature':temperature}
      f = open(setting_file_name, "w+", encoding='UTF-8')
      json.dump(setting_record, f, indent=4)
      f.close()

      file_name = f'./gen_results/random_answers_{signed_k}shot_{num_calls}calls_{top_starts}_{tail_starts}_{retriever_name}_dl_{dataset_name}_prompt1.json'
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
      
      print(existed_qids)

      q_no = 0
      for qid, query in zip(queries['qid'].tolist(), queries['query'].tolist()):
            
            print(f'q_number={q_no}--{qid}')
            q_no += 1
            
            if(str(qid) not in existed_qids_list):
                  varying_context_result = {} #{start: results}
                  # existing_starts = []
            else:
                  # varying_context_result = result_to_write[str(qid)] #added 0824
                  # existing_starts = list(varying_context_result.keys()) #added 0824
                  continue
                  # print(f'q_number={q_no}--{qid}')
                  # print(existing_starts)

            start_records, context_book = prompt_tools.compose_context(qid=qid, res=res, k=k, step=step, top_starts=top_starts, tail_starts=tail_starts, doc_dict=doc_dict, reverse_order=reverse_order)
            for start, context in zip(start_records, context_book):
                  llm.set_seed(1000) # added 0824
                  # if(str(start) in existing_starts):
                  #       continue
                  print(f'\tstart_rank.{start}')
                  prompt = prompt_tools.prompt_assembler(context, query)
                  print(prompt)
                  multi_call_results = {}
                  for j in range(num_calls):
                        print(f'\t\tno.{j}')
                        result = llama_tools.single_call(llm=llm, prompt=prompt, temperature=temperature)
                        multi_call_results.update({j: result})
                  varying_context_result.update({start: multi_call_results})
                        
            result_to_write.update({qid: varying_context_result})              
            experiment_tools.update_json_result_file(file_name=file_name, result_to_write=result_to_write)