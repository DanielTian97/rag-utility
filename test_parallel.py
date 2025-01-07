import ctypes
import llama_cpp

llama_cpp.llama_backend_init(numa=False)

params = llama_cpp.llama_model_default_params()
params.n_gpu_layers = 35
model = llama_cpp.llama_load_model_from_file(
    b"../gguf_storage/Meta-Llama-3-8B-Instruct.Q8_0.gguf", params
)

n_ctx = 512
n_len = 32
n_parallel = 2
prompt = b"The quick brown fox"

tokens = (llama_cpp.llama_token * n_ctx)()
tokens_len = llama_cpp.llama_tokenize(
    model, prompt, len(prompt), tokens, len(tokens), True, True
)
print(tokens[:tokens_len])

n_kv_req = tokens_len + (n_len - tokens_len) * n_parallel
print(n_kv_req)

ctx_params = llama_cpp.llama_context_default_params()
ctx_params.seed = 1234
ctx_params.n_ctx = n_kv_req
ctx_params.n_batch = max(n_len, n_parallel)
ctx_params.n_threads = 1
ctx_params.n_threads_batch = 1
ctx = llama_cpp.llama_new_context_with_model(model, ctx_params)

n_ctx = llama_cpp.llama_n_ctx(ctx)
batch = llama_cpp.llama_batch_init(max(tokens_len, n_parallel), 0, 1)

batch.n_tokens = tokens_len
for i in range(tokens_len):
    batch.token[i] = tokens[i]
    batch.pos[i] = i
    batch.seq_id[i][0] = 0
    batch.n_seq_id[i] = 1
    batch.logits[i] = False

batch.logits[batch.n_tokens - 1] = True

if llama_cpp.llama_decode(ctx, batch) != 0:
    print("Error decoding")    

for i in range(n_parallel):
    llama_cpp.llama_kv_cache_seq_cp(ctx, 0, i, 0, batch.n_tokens)
    
# Initialize sampler chain with default parameters
sparams = llama_cpp.llama_sampler_chain_default_params()
sampler_chain = llama_cpp.llama_sampler_chain_init(sparams)

# Add top_k, top_p, temperature, and final distribution-based sampler
llama_cpp.llama_sampler_chain_add(sampler_chain, llama_cpp.llama_sampler_init_top_k(40))
llama_cpp.llama_sampler_chain_add(sampler_chain, llama_cpp.llama_sampler_init_top_p(0.9, 1))
llama_cpp.llama_sampler_chain_add(sampler_chain, llama_cpp.llama_sampler_init_temp(0.4))
llama_cpp.llama_sampler_chain_add(sampler_chain, llama_cpp.llama_sampler_init_dist(1234))  # Final "dist" sampler

streams = [""] * n_parallel
i_batch = [batch.n_tokens - 1] * n_parallel

n_cur = batch.n_tokens
n_decode = 0

while n_cur <= n_len:
    batch.n_tokens = 0
    for i in range(n_parallel):
        if i_batch[i] < 0:
            continue

        # Sample the next token using the sampler chain
        new_token_id = llama_cpp.llama_sampler_sample(sampler_chain, ctx, -1)

        if new_token_id == llama_cpp.llama_token_eos(ctx) or n_cur == n_len:
            i_batch[i] = -1
            continue

        buf = (ctypes.c_char * 32)()
        
        # Convert token ID to text
        outlen = llama_cpp.llama_token_to_piece(model, new_token_id, buf, len(buf), 0, False)
        streams[i] += bytes(buf[:outlen]).decode("utf-8")

        batch.token[batch.n_tokens] = new_token_id
        batch.pos[batch.n_tokens] = n_cur
        batch.seq_id[batch.n_tokens][0] = i
        batch.n_seq_id[batch.n_tokens] = 1
        batch.logits[batch.n_tokens] = True

        i_batch[i] = batch.n_tokens
        batch.n_tokens += 1
        n_decode += 1

    if batch.n_tokens == 0:
        break

    n_cur += 1

    if llama_cpp.llama_decode(ctx, batch) != 0:
        print("Error decoding", flush=True)
        break
    print(n_cur)
    print(streams)
    
print(streams)
llama_cpp.llama_batch_free(batch)
llama_cpp.llama_free(ctx)
llama_cpp.llama_free_model(model)
llama_cpp.llama_backend_free()