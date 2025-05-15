from typing import List, Literal
import string
import regex
from collections import Counter

def normalize_answer(s : str) -> str:
    def remove_articles(text: str) -> str:
        return regex.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())
    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match_score(prediction : str, ground_truth : str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def cal_em(prediction : str, ground_truths : List[str]) -> float:
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

def f1_score(prediction : str, ground_truth : str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO_METRIC = (0, 0, 0)
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC[0]
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC[0]
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC[0]
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # return f1, precision, recall
    return f1

def cal_f1(prediction : str, ground_truths : List[str]) -> float:
    return max([f1_score(prediction, gt) for gt in ground_truths])