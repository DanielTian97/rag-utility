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

To create a conda environment, simply use the file rag-utility.yml

**prepare ingredients for generation**

1. The query file. We have several files for the query sets, including (GROUP 1: for MS MARCO Passage) TREC-DL-19, 20, dev-small; (GROUP 2: for MS MARCO Passage v2) TREC-DL-21, 22; (GROUP 3 for Wiki) NQ test queries.
2. The res file. They should be .csv files with column names: qid, docid, docno, rank, score, query. This is the standard form of the results in PyTerrier: https://pyterrier.readthedocs.io/en/latest/transformer.html. 
3. The .pkl file for storing the diction of docid:doc_text. It is used to compose the contexts in RAG. You can use the script in doc_dicts directory to create them for MS MARCO Passage and MS MARCO PASSAGE v2.

### :large_blue_diamond: Step 1: Generation

1. **Basic generations Generation:**
   There are two ways for generation:
   (1) k-shot (k>=1): e.g. python rag-random_kshot.py --k 5 --dataset_name 19 --retriever bm25
   
       <k> is the number of documents in the context;\
       <dataset_name> is the query set to be tested (we support (trec-dl-) 19, 20, 21, 22; (MS MARCO PASSAGE dataset's) dev-small and Natural Question's nq_test;\
       <retriever> is the ranking model which retrieved the context (the valid ones are BM25, BM25>>MonoT5 and TCT_ColBERT).
   
      Besides, there are additional arguments for controlling the generation:
   
       <num_calls>: we support random answer generating based on the same prompt multiple times. The default value is 5.\
       <temperature>: controls the randomness of the generation process (default value is 0.3).\
       <step> <top> <tail>: for the same query, we support generation with context documents sampled from the top and tail of the provided res file. With a step of <step>, the program sample <top>-times from the top, and <tail> times from the end. The results will be distinguished with different identifiers showing the starting rank. Default setting is <step>=1, <top>=1, <tail>=0.\
       (still in the test stage) <long_answer>: controls the answer length by switching between the prompts for long-answer-generation and short-answer-generation (a phrase). The default is True, which will lead to generating long-form answers.
   
   (2) 0-shot: e.g. python rag-random_0shot.py --dataset_name 20
      Other settings besides <dataset_name> are <num_calls> <temperature> and <long_answers>, the same as for k-shot answers.
         
   All the res files are stored under ./res directory. <batch_size> \<step> <num_from_top> and <num_from_end> control the way of selecting the contexts from the res files.
   After generation, the results will be stored in ./middle_products directory with a setting_file which records the parameters.
   Note: the product file will be suffixed by '_prompt1' because it is in fact the second prompt that has been massively experimented.

3. **Generation with shuffled (permutated) contexts:**

### :large_blue_diamond: Step 2: Evaluation

1. To evaluate: python evaluation_result_full_new_prompt.py 3 5 19 10 0 _prompt1 mt5

   (The function takes 7 parameters corresponding to the product file which contains the generation results: <batch_size> <num_sample> <dataset_id> <num_from_top> <num_from_end> <prompt_suffix> <retriever>.
   The evaluation results will be stored in ./eval_results directory.)
