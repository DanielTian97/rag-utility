import requests
import sseclient
import json

def llama_call(llm, prompt, temperature, long_answer=True):
    
      llm.set_seed(1000)
      token_limit = 300
      stop_at = ["STOP"] if long_answer else ["</answer>"]
      
      output = llm(
                  prompt, # Prompt
                  max_tokens=token_limit, # Generate up to 300 tokens, set to None to generate up to the end of the context window
                  stop=stop_at, # Stop generating just before the model would generate a new question
                  echo=False, # Echo the prompt back in the output
                  logprobs=50,
                  top_k=50,
                  temperature=temperature,
            ) # Generate a completion, can also call create_completion
      
      return output

def llm_client(prompt, temperature, long_answer=True, port=8080):

    token_limit = 300
    stop_at = ["STOP"] if long_answer else ["</answer>"]

    url = f"http://localhost:{port}/v1/completions"
    # print(port)
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "local",
        "seed": 1000,
        "prompt": prompt,
        "max_tokens": token_limit,
        "temperature": temperature,
        "logprobs": 50,
        "top_k": 50,
        "stop": stop_at,
    }
    
    resp = requests.post(url, headers=headers, json=data, stream=True)
    client = sseclient.SSEClient(resp)
    output = resp.json()
      
    return output
  
def load_llama(model_path="../gguf_storage/Meta-Llama-3-8B-Instruct.Q8_0.gguf", load_on_which_gpu=0):
    
    from llama_cpp import Llama, LLAMA_SPLIT_MODE_NONE
    import torch
    print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
      
    llm = Llama(
        model_path=model_path,
        logits_all=True,
        verbose=False,
        n_gpu_layers=-1, # Uncomment to use GPU acceleration
        n_ctx=4000, # temporarily change to 4000
        main_gpu = load_on_which_gpu,
        split_mode=LLAMA_SPLIT_MODE_NONE,
    )

    llm.set_seed(1000)
    return llm
  
def single_call(llm, prompt, temperature, long_answer=True, mode='local', port=8080):

    if(mode=='local'):
        output = llama_call(llm, prompt, temperature, long_answer)           
        token_logprobs = output['choices'][0]['logprobs']['token_logprobs']   
        
    else: # mode=='server'
        output = llm_client(prompt, temperature, long_answer, port)
        try:
            token_logprobs = [i['top_logprobs'][0]['logprob'] for i in output["choices"][0]['logprobs']['content']]
        except:
            token_logprobs = []
    
    prob_seq = sum(token_logprobs)
    answer = output["choices"][0]["text"]
    result = {"answer": answer, "prob_seq": float(prob_seq), "probs": str(token_logprobs)}
        
    return result

def testtesttest():
    print('1111')