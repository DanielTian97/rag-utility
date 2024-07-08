from llama_cpp import Llama
from compose_prompts import *

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

for query in query_list:
      
      preamble = "Please answer this question. End your answer with STOP."
      prompt = f'{preamble} Question: \'{query}\' \nAnswer: '
      print(prompt)

      output = llm(
            prompt, # Prompt
            max_tokens=300, # Generate up to 300 tokens, set to None to generate up to the end of the context window
            stop=["STOP"], # Stop generating just before the model would generate a new question
            echo=False, # Echo the prompt back in the output
            # logprobs=1,
            temperature=0,
            # top_k = 3,
      ) # Generate a completion, can also call create_completion

      print(output)
      # print(output.keys())
  
    #   top_logits = output['choices'][0]['logprobs']['top_logprobs'][-1]