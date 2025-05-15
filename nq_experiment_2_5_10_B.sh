python rag-random_kshot.py --k 2 --dataset_name nq_test --retriever mt5 --temperature 0 --num_calls 1 --long_answer False
python rag-random_kshot.py --k 5 --dataset_name nq_test --retriever mt5 --temperature 0 --num_calls 1 --long_answer False
python rag-random_kshot.py --k 10 --dataset_name nq_test --retriever mt5 --temperature 0 --num_calls 1 --long_answer False

python rag-random_kshot.py --k 5 --dataset_name nq_test --retriever tct --temperature 0 --num_calls 1 --long_answer False
python rag-random_kshot.py --k 10 --dataset_name nq_test --retriever tct --temperature 0 --num_calls 1 --long_answer False