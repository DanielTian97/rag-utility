from llama_cpp import Llama
from compose_prompts import *
import json
import sys

def llama_call(llm, prompt):
      
      output = llm(
                  prompt, # Prompt
                  max_tokens=300, # Generate up to 300 tokens, set to None to generate up to the end of the context window
                  stop=["STOP"], # Stop generating just before the model would generate a new question
                  echo=False, # Echo the prompt back in the output
                  logprobs=50,
                  top_k=50,
                  temperature=0.3,
            ) # Generate a completion, can also call create_completion
      
      return output

def update_json_result_file(file_name, result_to_write):
      f = open(file_name, "w+", encoding='UTF-8')
      json.dump(result_to_write, f, indent=4)
      f.close()

if __name__=="__main__":
      num_calls = int(sys.argv[1])
      dataset_name = str(sys.argv[2])

      file_name = f'./middle_products/random_answers_0shot_{num_calls}calls_dl_{dataset_name}.json'

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

      llm = Llama(
            model_path="../Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct.Q8_0.gguf",
            logits_all=True,
            verbose=False,
            n_gpu_layers=-1, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            # n_ctx=2048, # Uncomment to increase the context window
      )

      query = ''

      queries = pd.read_csv(f'./middle_products/queries_{dataset_name}.csv')
      qid_list = queries['qid'].tolist()
      query_list = queries['query'].tolist()

      file_name = f'./middle_products/random_answers_0shot_{num_calls}calls_dl_{dataset_name}.json'

      q_no = 0
      for qid, query in zip(qid_list, query_list):
            answers_0shot = {}
            result_to_write.update({qid: {0: answers_0shot}})
            llm.set_seed(1000)
            print(f'{q_no} {qid}')
            q_no += 1
            
            preamble = "Please answer this question. End your answer with STOP."
            prompt = f'{preamble} Question: \'{query}\' \nAnswer: '
            print(prompt)
            
            for i in range(num_calls):
                  print(f'no.{i}')
                  output = llama_call(llm, prompt)
                  logprob_dict = output['choices'][0]['logprobs']['top_logprobs']

                  token_logprobs = output['choices'][0]['logprobs']['token_logprobs']
                  prob_seq = sum(token_logprobs)
                  
                  answer = output['choices'][0]['text']
                  answers_0shot.update({i: {"answer": answer, "prob_seq": float(prob_seq)}})
            
            update_json_result_file(file_name=file_name, result_to_write=result_to_write)