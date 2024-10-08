def llama_call(llm, prompt, temperature):
      
      output = llm(
                  prompt, # Prompt
                  max_tokens=300, # Generate up to 300 tokens, set to None to generate up to the end of the context window
                  stop=["STOP"], # Stop generating just before the model would generate a new question
                  echo=False, # Echo the prompt back in the output
                  logprobs=50,
                  top_k=50,
                  temperature=temperature,
            ) # Generate a completion, can also call create_completion
      
      return output
  
def load_llama():
    
    from llama_cpp import Llama
    import torch
    print(torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
      
    llm = Llama(
        model_path="../Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct.Q8_0.gguf",
        logits_all=True,
        verbose=False,
        n_gpu_layers=-1, # Uncomment to use GPU acceleration
        n_ctx=2048, # Uncomment to increase the context window
        # n_ctx=3000, # temporarily change to 3000
    )

    llm.set_seed(1000)
    return llm
  
def single_call(llm, prompt, temperature):
    output = llama_call(llm, prompt, temperature)
                  
    # logprob_dict = output['choices'][0]['logprobs']['top_logprobs']
    token_logprobs = output['choices'][0]['logprobs']['token_logprobs']
    prob_seq = sum(token_logprobs)
                  
    answer = output['choices'][0]['text']
                  
    result = {"answer": answer, "prob_seq": float(prob_seq)}
    return result

def testtesttest():
    print('1111')