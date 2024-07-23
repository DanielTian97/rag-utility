from llama_cpp import Llama
from compose_prompts import *

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

def select_passages(res, qid: str):
    retrieved_num = res[res.qid==qid]['rank'].max()
    print(retrieved_num)
    print(res[(res.qid==qid)&(res['rank']<100)].head(10))
    
def compose_context(res, qid: str):
      

llm = Llama(
      model_path="../Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct.Q8_0.gguf",
      logits_all=True,
      verbose=False,
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)

llm.set_seed(1000)

query = ''

queries = pd.read_csv('./middle_products/queries_19.csv')
qid_list = queries['qid'].tolist()
query_list = queries['query'].tolist()
res = pd.read_csv('./res/bm25_dl_19.csv')

file_name = './middle_products/random_answers.txt'
f = open(file_name, "w+", encoding='UTF-8')

q_no = 0
for qid, query in zip(qid_list, query_list):
      print(f'{q_no} {qid}')
      q_no += 1
      f.write(f'---QUERY---{qid}\t{query}\n')
      
      preamble = "Please answer this question based on the given context. End your answer with STOP."
      context_book = compose_context(qid)
      for context in context_book:
            prompt = f'{preamble} Context: {context} \nQuestion: \'{query}\' \nAnswer: '
            print(prompt)
            
            for i in range(5):
                  print(f'no.{i}')
                  output = llama_call(llm, prompt)
                  logprob_dict = output['choices'][0]['logprobs']['top_logprobs']
                  # print(len(logprob_dict))
                  token_logprobs = output['choices'][0]['logprobs']['token_logprobs']
                  prob_seq = sum(token_logprobs)
                  
                  answer = output['choices'][0]['text']
                  to_write = f'{answer}\nPROB_LOG:{prob_seq}\n'
                  f.write(to_write)
    
f.close()