import json
import os
from bert_score import score

def load_results(file_path):
    pred = []
    gt = []
    with open(file_path, 'r') as file:
        for line in file:
            result = json.loads(line)
            pred.append(result['generated_answer'])
            gt.append(result['reference_answer'])
    return pred, gt

def bertScore(pred, gt):
    P, R, F1 = score(pred, gt, lang="en", verbose=True)
    precision = P.mean().item()
    recall = R.mean().item()
    f1 = F1.mean().item()
    return precision, recall, f1

if __name__ == "__main__":
    result_file_path =  r'..\output\GSM8K_result.jsonl'
    pred, gt = load_results(result_file_path)
    precision, recall, f1 = bertScore(pred, gt)

    print(f"BERTScore Precision: {precision:.4f}")
    print(f"BERTScore Recall: {recall:.4f}")
    print(f"BERTScore F1: {f1:.4f}")

# BERTScore Precision: 0.7982
# BERTScore Recall: 0.8506
# BERTScore F1: 0.8230