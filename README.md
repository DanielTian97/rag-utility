# An Experiment Framework for Evaluating the Effect of Varying Contexts on the RAG Performances in Downstream Tasks

:writing_hand: This project was originally built to investigate how relevance is transmitted from the retriever to the generator. The first-stage outcomes and analysis of the correlation between conventional IR metrics and utility (relative gain in RAG performance) have been reported in our full paper, which has been accepted by ECIR 2025. Currently, this paper can be accessed via .... on arXiv.

:dart: In this project, we built a straightforward RAG pipeline, which takes retrieved passages as context to facilitate the generation of LLM. To be specific, the downstream task of the generation is Question Answering (QA). Regarding the aim of our study, for which this repository was developed, we use MS-MARCO PASSAGE as the database for retrieving the context, while the test queries in TREC DL-19/DL-20 and MS-MARCO dev set (small) are the questions in the QA task. In this way, the relevance of retrieved context can be well assessed by conventional IR metrics like nDCG@*k*. The adoption of IR queries in QA tasks causes a lack of reference in evaluating the generated answers, which is addressed in this code by extending the evaluation framework proposed by .. and .. in ... . The denoted relevant documents in qrels files are used as pseudo references. BERTScore analyses this similarity in our code.

:sparkles: The aforementioned framework brings an inherent benefit for the researcher who wants to gain a deeper understanding of RAG workflow: both the :one: retrieved context and :two: generated context are analysed based on the same set of references. Therefore, the metrics calculated about the input .... 

## How to use it?

:footprints: There are two steps, namely, generating answers and evaluating the answers; we provide example commands for each of them in the later paragraphs. Before commencing the experiments, there need some preparations for setting up the environment.

### :large_blue_diamond: Step 0: Preparations

**Downloading the quantised model and necessary files:** This code is based on the library llama-cpp-python. To correctly load the model from the .GGUF, please put the project under a directory where contains a sub-directory named 
gguf_storage (with Meta-Llama-3-8B-Instruct.Q8_0.gguf in it, download link: https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF). Please note, the folder named gguf_storage should be outside the folder which is cloned from this repository.

**Creating a conda environment:**

### :large_blue_diamond: Step 1: Generation

1. **Basic generations Generation:**
   There are two ways for generation: python rag-random_kshot_prompt1.py 3 1 5 10 0 0.3 19 mt5

   (The function takes 8 parameters: <batch_size> \<step> <num_sample> <num_from_top> <num_from_end> \<temperature> <dataset_identication> \<retriever>, the default retriever is bm25.
   All the res files are stored under ./res directory. <batch_size> \<step> <num_from_top> and <num_from_end> control the way of selecting the contexts from the res files.
   After generation, the results will be stored in ./middle_products directory with a setting_file which records the parameters.
   Note: the product file will be suffixed by '_prompt1' because it is in fact the second prompt that has been massively experimented.)

2. **Generation with shuffled (permutated) contexts:**

### :large_blue_diamond: Step 2: Evaluation

1. To evaluate: python evaluation_result_full_new_prompt.py 3 5 19 10 0 _prompt1 mt5

   (The function takes 7 parameters corresponding to the product file which contains the generation results: <batch_size> <num_sample> <dataset_id> <num_from_top> <num_from_end> <prompt_suffix> <retriever>.
   The evaluation results will be stored in ./eval_results directory.)
