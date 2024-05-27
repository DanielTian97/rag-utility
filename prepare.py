from llama_cpp import Llama

llm = Llama(
      model_path="../Meta-Llama-3-8B/Meta-Llama-3-8B.Q8_0.gguf",
      logits_all=True,
      verbose=False,
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)


prompt = 'Is spruce a species of tree?'
output = llm(
      # f"Q: {prompt} Briefly summarise your reason and answer 'Yes' or 'No'. A: ", # Prompt
      # f"Q: {prompt} Answer 'Yes' or 'No'. A: ", # Prompt
      f"Q: {prompt} 'Yes' or 'No', and then summarise your reason. A: ", # Prompt
      max_tokens=1, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Yes", "No"], # Stop generating just before the model would generate a new question
      # echo=True, # Echo the prompt back in the output
      logprobs=60000,
      temperature=0,
) # Generate a completion, can also call create_completion

# print(output)
# print(output.keys())
print(prompt)
top_logits = output['choices'][0]['logprobs']['top_logprobs'][-1]
print(top_logits['Yes'])
print(top_logits['No'])