from llama_cpp import Llama
from compose_prompts import *
from tqdm import tqdm

WIDTH = 3
TOP_K = 3

def llama_call(llm, prompt):
      
      output = llm(
                  prompt, # Prompt
                  max_tokens=1, # Generate up to 32 tokens, set to None to generate up to the end of the context window
                  # stop=["STOP"], # Stop generating just before the model would generate a new question
                  echo=False, # Echo the prompt back in the output
                  logprobs=WIDTH,
                  temperature=0,
            ) # Generate a completion, can also call create_completion
      
      return output

def sort_dict(target: dict, stop_token_prob: dict):
      sorted_dict = dict(sorted(target.items(), key=lambda item: -item[1])[:TOP_K])
      
      for seq_1 in stop_token_prob.keys():
            if seq_1 in sorted_dict.keys():
                  clean_prob = sorted_dict[seq_1] - stop_token_prob[seq_1]
                  sorted_dict.update({seq_1: clean_prob})
                  
      return dict(sorted(sorted_dict.items(), key=lambda item: -item[1])[:TOP_K])

llm = Llama(
      model_path="../Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct.Q8_0.gguf",
      logits_all=True,
      verbose=False,
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)

query = ''

queries = pd.read_csv('./middle_products/queries_19.csv')
query_list = queries['query'].tolist()

for query in query_list[:1]:
      
      preamble = "Please answer this question. End your answer with STOP."
      prompt = f'{preamble} Question: \'{query}\' \nAnswer: '
      print(prompt)
      
      top_logits_seq = {"": 0}
      
      stopped_seq = []
      

      for i in tqdm(range(300)):
            temp_dict = {}
            stop_token_prob = {}
            
            skipped = 0
            for seq in top_logits_seq:
                  if(seq in stopped_seq):
                        temp_dict.update({seq: top_logits_seq[seq]})
                        skipped += 1
                        continue
                  
                  prompt_added = f'{prompt}{seq}'
                  
                  output = llama_call(llm, prompt_added)
                  
                  top_logits_token = output['choices'][0]['logprobs']['top_logprobs'][-1]
                  # print(top_logits_token)
                  
                  for token in top_logits_token:
                        if(token==" STOP"):
                              print(top_logits_token)
                              stopped_seq.append(seq)
                              stop_token_prob.update({seq: top_logits_token[token]})
                              temp_dict.update({seq: top_logits_seq[seq]+top_logits_token[token]})
                              
                              print(stop_token_prob[seq], temp_dict[seq])
                              print("!!")
                        else:
                              temp_dict.update({f'{seq}{token}': top_logits_seq[seq]+top_logits_token[token]})
            
            top_logits_seq = sort_dict(temp_dict, stop_token_prob)
            
            if(skipped == TOP_K):
                  break
      
      print(top_logits_seq)