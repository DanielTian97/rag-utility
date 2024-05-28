from llama_cpp import Llama
from compose_prompts import *
#pip install llama-cpp-python
import torch
from tqdm import tqdm
import sys

def prepare_model():
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      
      print('preparing model ......')
      if(device != "cpu"):
            llm = Llama(
                  model_path="../Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct.Q8_0.gguf",
                  logits_all=True,
                  verbose=False,
                  n_gpu_layers=-1, # Uncomment to use GPU acceleration
                  seed=1337, # Uncomment to set a specific seed
                  # n_ctx=2048, # Uncomment to increase the context window
            )
      else:
            llm = Llama(
                  model_path="../Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct.Q8_0.gguf",
                  logits_all=True,
                  verbose=False,
                  # n_gpu_layers=-1, # Uncomment to use GPU acceleration
                  seed=1337, # Uncomment to set a specific seed
                  # n_ctx=2048, # Uncomment to increase the context window
            )
      return llm

def prepare_pairs(dataset: str):

      print('read query document pairs')
      query = ''
      document = ''
      q_d_pair_list = get_msmarco_passage_pairs(dataset)
      return q_d_pair_list

def making_predictions(llm, q_d_pair_list):
      list_yes = ['Yes', ' Yes', 'yes', ' yes', 'YES', ' YES']
      list_no = ['No', ' No', 'no', ' no', 'NO', ' NO']

      print('start predicting')
      for i in tqdm(range(len(q_d_pair_list))):
            
            query = q_d_pair_list[i].qText
            document = q_d_pair_list[i].dText
            question = f'Is the provided passage relevant to the provided query?'


            prompt = f"Query: {query}\nPassage: {document}\nQuestion: {question}\nAnswer: " #use it
            output = llm(
                  prompt, # Prompt
                  max_tokens=1, # Generate up to 32 tokens, set to None to generate up to the end of the context window
                  stop=[" Yes", " No", "Yes", "No"], # Stop generating just before the model would generate a new question
                  echo=False, # Echo the prompt back in the output
                  logprobs=60000,
                  temperature=0,
            ) # Generate a completion, can also call create_completion


            top_logits = output['choices'][0]['logprobs']['top_logprobs'][-1]
            
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

            # print(prompt)
            # print('Yes', logit_yes)
            # print('No', logit_no)
            # print(logit_yes - logit_no)
      
            pred = (logit_yes - logit_no)
            q_d_pair_list[i].put_prediction(pred)
      return q_d_pair_list

def save_results(q_d_pair_list, dataset):            
      print('save results')
      import pickle

      with open(f'prediction_record_{dataset}.pkl', 'wb') as f:
            pickle.dump(q_d_pair_list, f)

def experiment(dataset: str):
      llm = prepare_model()
      q_d_pair_list = prepare_pairs(dataset)
      q_d_pair_list = making_predictions(llm, q_d_pair_list)
      save_results(q_d_pair_list, dataset)
    
if __name__=="__main__":
    dataset = str(sys.argv[1])
    experiment(dataset)