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

example_pair_book = {}

examples = 'Oak is a species of tree. Egg is not a species of tree.'

query = ''
document = ''
q_d_pair_list = get_msmarco_passage_pairs()



list_yes = ['Yes', ' Yes', 'yes', ' yes', 'YES', ' YES']
list_no = ['No', ' No', 'no', ' no', 'NO', ' NO']

for i in range(10):
      
      query = q_d_pair_list[i].qText
      document = q_d_pair_list[i].dText
      question = f'Is passage \'{document}\' relevant to the query \'{query}\'?'
      print(question)

      # prompt = f"Examples: {examples} \nQuestion: {question} Answer Yes or No. \nAnswer: " #use it
      # prompt = f"Examples: {examples} \nQuestion: {question} Answer Yes or No, and then summarise your reason. \nAnswer: "
      prompt = f"Question: {question} Answer Yes or No. \nAnswer: " #use it
      # prompt = f"Question: {question} Briefly summarise your reason and answer 'Yes' or 'No'. \nAnswer: "
      output = llm(
            prompt, # Prompt
            max_tokens=1, # Generate up to 32 tokens, set to None to generate up to the end of the context window
            stop=[" Yes", " No", "Yes", "No"], # Stop generating just before the model would generate a new question
            echo=True, # Echo the prompt back in the output
            logprobs=60000,
            temperature=0,
      ) # Generate a completion, can also call create_completion

      # print(output)
      # print(output.keys())
      print(prompt)
      top_logits = output['choices'][0]['logprobs']['top_logprobs'][-1]
      # print(top_logits)
      
      logit_yes = top_logits[list_yes[0]]
      for word in list_yes[1:]:
            if(word in top_logits.keys()):
                  if(top_logits[word] > logit_yes):
                        logit_yes = top_logits[word]
                  
      logit_no = top_logits[list_no[0]]
      for word in list_no[1:]:
            if(word in top_logits.keys()):
                  if(top_logits[word] > logit_no):
                        logit_no = top_logits[word]

      print('Yes', logit_yes)
      print('No', logit_no)
      print(logit_yes - logit_no)