import os
import sys
import csv
import statistics
from typing import List, Dict

# Ensure project root is on `sys.path` so imports from repository root work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rag_pipeline import answer_question, retrieve_chunks_with_confidence, embed_model
from verification.hallucination_detector import compute_hallucination_score
from verification.sentence_coverage import sentence_coverage_ratio
from verification.concept_overlap import concept_overlap_score
from verification.final_trust import final_trust_score


def embed_cosine(a_vec, b_vec):
    a = a_vec / (np.linalg.norm(a_vec) + 1e-12)
    b = b_vec / (np.linalg.norm(b_vec) + 1e-12)
    return float(np.dot(a, b))


def evaluate_one(question: str, gold: str) -> Dict:
    chunks, retrieval_conf = retrieve_chunks_with_confidence(question)
    our = answer_question(question)
    answer = our.get("answer", "")

    # embed gold + answer
    emb = embed_model.encode([gold, answer])
    gold_emb, ans_emb = emb[0], emb[1]

    acc_sim = embed_cosine(gold_emb, ans_emb)
    halluc = compute_hallucination_score(answer, chunks)
    coverage = sentence_coverage_ratio(answer, chunks)
    overlap = concept_overlap_score(answer, chunks)
    final_tr = final_trust_score(halluc, coverage, overlap, retrieval_conf)

    return {
        "question": question,
        "gold": gold,
        "answer": answer,
        "acc_sim": acc_sim,
        "hallucination": halluc,
        "sentence_coverage": coverage,
        "concept_overlap": overlap,
        "retrieval_confidence": retrieval_conf,
        "final_trust": final_tr,
    }


def read_benchmarks(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmarks file not found: {path}")
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = r.get('question') or r.get('Question')
            g = r.get('gold_answer') or r.get('gold') or r.get('Gold')
            if q and g:
                rows.append({'question': q, 'gold': g})
    return rows


def main():
    os.makedirs('evaluation', exist_ok=True)
    bench_path = os.path.join('benchmarks', 'benchmarks.csv')
    try:
        items = read_benchmarks(bench_path)
    except FileNotFoundError as e:
        print(e)
        print('Create a CSV at benchmarks/benchmarks.csv with columns: question,gold_answer')
        return

    results = []
    for it in items:
        print('Evaluating:', it['question'])
        r = evaluate_one(it['question'], it['gold'])
        results.append(r)

    out_csv = os.path.join('evaluation', 'results_rag.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['question', 'gold', 'answer', 'acc_sim', 'hallucination', 'sentence_coverage', 'concept_overlap', 'retrieval_confidence', 'final_trust'])
        for r in results:
            writer.writerow([r['question'], r['gold'], r['answer'], r['acc_sim'], r['hallucination'], r['sentence_coverage'], r['concept_overlap'], r['retrieval_confidence'], r['final_trust']])

    # summary
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.mean(xs) if xs else 0.0

    summary = {
        'mean_acc_sim': mean([r['acc_sim'] for r in results]),
        'mean_hallucination': mean([r['hallucination'] for r in results]),
        'mean_sentence_coverage': mean([r['sentence_coverage'] for r in results]),
        'mean_concept_overlap': mean([r['concept_overlap'] for r in results]),
        'mean_retrieval_confidence': mean([r['retrieval_confidence'] for r in results]),
        'mean_final_trust': mean([r['final_trust'] for r in results])
    }

    summary_path = os.path.join('evaluation', 'summary_rag.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for k, v in summary.items():
            writer.writerow([k, v])

    print('Done. Results ->', out_csv, ' Summary ->', summary_path)


if __name__ == '__main__':
    main()
