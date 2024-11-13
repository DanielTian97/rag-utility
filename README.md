### A project to investigate the effect of contexts on the RAG performance.

In this project, we built a simple RAG pipeline, which takes retrieved raw passages as context to LLM.

#### How to use it?

There are two steps, namely, generating answers and evaluating the answers, here are example commands for each of them:

1. To generate: python rag-random_kshot_prompt1.py 3 1 5 10 0 0.3 19 mt5

   (The function takes 8 parameters: <batch_size> <step_size> <num_sample> <num_from_top> <num_from_end> <temperature> <dataset_identication> <retriever>, the default retriever is bm25.
   All the res files are stored under ./res directory. <batch_size> <step_size> <num_from_top> and <num_from_end> control the way of selecting the contexts from the res files.
   After generation, the results will be stored in ./middle_products directory with a setting_file which records the parameters.
   Note: the product file will be suffixed by '_prompt1' because it is in fact the second prompt that has been massively experimented.)
3. To evaluate: python evaluation_result_full_new_prompt.py 3 5 19 10 0 _prompt1 mt5

   (The function takes 7 parameters corresponding to the product file which contains the generation results: <batch_size> <num_sample> <dataset_id> <num_from_top> <num_from_end> <prompt_suffix> <retriever>.
   The evaluation results will be stored in ./eval_results directory.)
