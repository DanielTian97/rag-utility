from tools import llama_tools, prompt_tools, experiment_tools
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
      parser.add_argument("--dataset_name", type=str, choices=['19', '20', '21', '22', 'dev_small', 'nq_test'])
      parser.add_argument("--retriever", type=str, default='bm25', choices=['bm25', 'mt5', 'tct', 'oracle', 'reverse_oracle'])
      parser.add_argument("--full_permutation", type=str, default='False', choices=['False', 'True'])
      args = parser.parse_args()

      k = args.k
      step = args.step
      num_calls = args.num_calls
      # start control parameters
      tops = args.tops
      tails = args.tails
      temperature = args.temperature
      dataset_name = args.dataset_name
      retriever_name = args.retriever
      full_permutation = eval(args.full_permutation)
      if(full_permutation):
            print("All possible permutations will be generated.")
      else:
            print("Only (1) origin; (2) reversed; (3) top_ranked-in-middle; (4) top_ranked-at-2-ends; will be generated.")
      
      # load the llm
      llm = llama_tools.load_llama()
      # load needed data
      doc_dict, queries, res = prompt_tools.prepare_data(dataset_name, retriever_name)
      
      setting_file_name = f'./gen_results/random_answers_{k}shot_{num_calls}calls_{tops}_{tails}_{retriever_name}_dl_{dataset_name}_settings_p_prompt1.json'
      setting_record = {'k':k, 'step':step, 'num_calls':num_calls, \
                  'tops':tops, 'tails':tails, 'temperature':temperature}
      f = open(setting_file_name, "w+", encoding='UTF-8')
      json.dump(setting_record, f, indent=4)
      f.close()

      file_name = f'./gen_results/random_answers_{k}shot_{num_calls}calls_{tops}_{tails}_{retriever_name}_dl_{dataset_name}_p_prompt1.json'
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
                  tops=tops, tails=tails, doc_dict=doc_dict, full_permutations=full_permutation)
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