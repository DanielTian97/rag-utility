python rag-random_kshot.py --dataset_name 19
python rag-random_kshot.py --k 5 --dataset_name 19 --retriever bm25
python rag-random_kshot_with_permutations.py --k 5 --dataset_name 19 --retriever bm25
python evaluator.py --k 5 --dataset_name 19 --retriever bm25
python evaluator.py --k 5 --dataset_name 19 --retriever bm25 --suffix _p