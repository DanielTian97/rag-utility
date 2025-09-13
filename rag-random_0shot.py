from tools import llama_tools, prompt_tools, experiment_tools
import json
import argparse
import datetime

if __name__=="__main__":
      
      parser = argparse.ArgumentParser()
      parser.add_argument("--num_calls", type=int, default=5)
      parser.add_argument("--temperature", type=float, default=0.3)
      parser.add_argument("--dataset_name", type=str, choices=['19', '20', '21', '22', 'dev_small', 'nq_test', 'nq_dev', 'hotpotqa_dev'])
      parser.add_argument("--long_answer", type=str, default='True', choices=['False', 'True'])
      args = parser.parse_args()

      k = 0
      step = 0
      num_calls = args.num_calls
      # start control parameters
      tops = 0
      tails = 0
      temperature = args.temperature
      dataset_name = args.dataset_name
      retriever_name = 'bm25'
      long_answer = True if args.long_answer=='True' else False
      short_answer_identifier = 'random_answers' if long_answer else 'short_answers'
      
      # load the llm
      llm = llama_tools.load_llama()
      # load needed data
      doc_dict, queries, res = prompt_tools.prepare_data(dataset_name)
      
      # setting_file_name = f'./gen_results/{short_answer_identifier}_answers_{k}shot_{num_calls}calls_{tops}_{tails}_{retriever_name}_dl_{dataset_name}_prompt1_settings.json'
      # setting_record = {'k': k, 'num_calls': num_calls, 'step': step, 'tops': tops, 'tails': tails, 
      #       'temperature': temperature, 'query_set': dataset_name, 'retriever': retriever_name, 'long_answer?': long_answer}
      # setting_record.update({'--experiment_start_at': str(datetime.datetime.now())})
      # f = open(setting_file_name, "w+", encoding='UTF-8')
      # json.dump(setting_record, f, indent=4)
      # f.close()

      if(long_answer):
          file_name = f'./gen_results/{short_answer_identifier}_{k}shot_{num_calls}calls_{tops}_{tails}_{retriever_name}_dl_{dataset_name}_prompt1.json'
      else:
          file_name = f'./gen_results/{short_answer_identifier}_{k}shot_{num_calls}calls_{tops}_{tails}_{retriever_name}_dl_{dataset_name}_concise.json'  # 0513
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

      q_no = 0
      for qid, query in zip(queries['qid'].tolist(), queries['query'].tolist()):
            
            print(f'q_number={q_no}--{qid}')
            q_no += 1
            
            if(str(qid) not in existed_qids_list):
                  varying_context_result = {} #{start: results}
            else:
                  continue

            zeroshot_result = {} #{start: results}

            llm.set_seed(1000) # added 0824

            prompt = prompt_tools.prompt_assembler_0(query, long_answer)
            print(prompt)
            multi_call_results = {}
            for j in range(num_calls):
                  print(f'\t\tno.{j}')
                  result = llama_tools.single_call(llm=llm, prompt=prompt, temperature=temperature, long_answer=long_answer)
                  multi_call_results.update({j: result})
            zeroshot_result.update({'0': multi_call_results})
                        
            result_to_write.update({qid: zeroshot_result})              
            experiment_tools.update_json_result_file(file_name=file_name, result_to_write=result_to_write)