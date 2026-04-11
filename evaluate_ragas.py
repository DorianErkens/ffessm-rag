"""
Évaluation RAGAS — comparaison côte à côte avec evaluate.py maison.

Réutilise retrieve() et generate() de evaluate.py sans les modifier.
Configure RAGAS avec Claude Haiku comme juge LLM + même SentenceTransformer.

Usage : python evaluate_ragas.py
"""
import os
import json
import asyncio
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import anthropic

load_dotenv()

# ─── Imports evaluate.py (sans modification) ─────────────────────────────────
from evaluate import (
    retrieve,
    generate,
    GOLDEN_DATASET,
    EMBEDDING_MODEL,
    INDEX_NAME,
    score_faithfulness,
    score_answer_relevancy,
    score_context_recall,
)

# ─── Imports RAGAS 0.4.x ─────────────────────────────────────────────────────
from ragas import EvaluationDataset, SingleTurnSample, evaluate as ragas_evaluate
from ragas.metrics.collections import (
    faithfulness as ragas_faithfulness,
    answer_relevancy as ragas_answer_relevancy,
    context_recall as ragas_context_recall,
    context_precision as ragas_context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings


def build_ragas_llm() -> LangchainLLMWrapper:
    """Claude Haiku comme juge LLM pour RAGAS — cohérent avec evaluate.py."""
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=512,
    )
    return LangchainLLMWrapper(llm)


def build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Même SentenceTransformer que l'ingestion — cohérence garantie."""
    emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return LangchainEmbeddingsWrapper(emb)


def run_maison_scores(samples: list[dict], claude) -> list[dict]:
    """
    Calcule les scores evaluate.py maison sur les mêmes samples.
    Retourne faithfulness, answer_relevancy, context_recall par question.
    """
    print("\n📐 Calcul des scores maison (evaluate.py)...")
    results = []
    for i, s in enumerate(samples):
        print(f"  [{i+1}/{len(samples)}] {s['question'][:55]}...")
        faith, _ = score_faithfulness(s["question"], s["answer"], s["contexts"], claude)
        rel, _ = score_answer_relevancy(s["question"], s["answer"], claude)
        recall, _ = score_context_recall(s["question"], s["contexts"], s["reference"], claude)
        results.append({
            "faithfulness": faith,
            "answer_relevancy": rel,
            "context_recall": recall,
        })
    return results


def run_ragas_scores(samples: list[dict], ragas_llm, ragas_emb) -> dict:
    """
    Calcule les 4 métriques RAGAS sur les mêmes samples.
    Retourne les scores moyens.
    """
    print("\n🔬 Calcul des scores RAGAS...")

    # Configure les métriques avec le LLM et les embeddings
    metrics = [ragas_faithfulness, ragas_answer_relevancy, ragas_context_recall, ragas_context_precision]
    for m in metrics:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_emb

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s["reference"],
        )
        for s in samples
    ])

    result = ragas_evaluate(dataset=dataset, metrics=metrics)
    return result.to_pandas().mean(numeric_only=True).to_dict()


def print_comparison(maison_scores: list[dict], ragas_scores: dict):
    """Tableau comparatif côte à côte."""
    # Moyennes maison
    avg_maison = {
        "faithfulness": sum(s["faithfulness"] for s in maison_scores) / len(maison_scores),
        "answer_relevancy": sum(s["answer_relevancy"] for s in maison_scores) / len(maison_scores),
        "context_recall": sum(s["context_recall"] for s in maison_scores) / len(maison_scores),
    }

    print("\n" + "═" * 65)
    print("COMPARAISON  evaluate.py maison  vs  RAGAS 0.4")
    print("═" * 65)
    print(f"{'Métrique':<25} {'Maison':>10} {'RAGAS':>10} {'Δ':>8}")
    print("─" * 65)

    pairs = [
        ("faithfulness",     "faithfulness",     "faithfulness"),
        ("answer_relevancy", "answer_relevancy",  "answer_relevancy"),
        ("context_recall",   "context_recall",    "context_recall"),
        ("context_precision","—",                 "context_precision"),
    ]

    for label, maison_key, ragas_key in pairs:
        maison_val = avg_maison.get(maison_key)
        ragas_val = ragas_scores.get(ragas_key)
        maison_str = f"{maison_val:.3f}" if maison_val is not None else "  —  "
        ragas_str  = f"{ragas_val:.3f}"  if ragas_val  is not None else "  —  "
        if maison_val is not None and ragas_val is not None:
            delta = ragas_val - maison_val
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "  —  "
        print(f"{label:<25} {maison_str:>10} {ragas_str:>10} {delta_str:>8}")

    print("═" * 65)
    print("""
Interprétation du Δ :
  Δ positif → RAGAS juge mieux que notre implem maison
  Δ négatif → notre implem maison est plus sévère
  Écart > 0.15 → divergence significative de définition de la métrique
""")


def main():
    print("🔤 Chargement des ressources...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(INDEX_NAME)
    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    ragas_llm = build_ragas_llm()
    ragas_emb = build_ragas_embeddings()

    # ─── Retrieval + génération (partagés entre les deux évals) ──────────────
    print(f"\n📥 Retrieval + génération sur {len(GOLDEN_DATASET)} questions...")
    samples = []
    for i, item in enumerate(GOLDEN_DATASET):
        print(f"  [{i+1}/{len(GOLDEN_DATASET)}] {item['question'][:55]}...")
        contexts = retrieve(item["question"], model, index)
        answer   = generate(item["question"], contexts, claude)
        samples.append({
            "question":  item["question"],
            "reference": item["reference"],
            "contexts":  contexts,
            "answer":    answer,
        })

    # ─── Scores maison ────────────────────────────────────────────────────────
    maison_scores = run_maison_scores(samples, claude)

    # ─── Scores RAGAS ─────────────────────────────────────────────────────────
    ragas_scores = run_ragas_scores(samples, ragas_llm, ragas_emb)

    # ─── Comparaison ──────────────────────────────────────────────────────────
    print_comparison(maison_scores, ragas_scores)

    # ─── Sauvegarde ───────────────────────────────────────────────────────────
    output = {
        "samples": [
            {**s, "contexts": s["contexts"][:2], "maison": maison_scores[i], "ragas": ragas_scores}
            for i, s in enumerate(samples)
        ],
        "averages": {
            "maison": {
                "faithfulness":    sum(s["faithfulness"]    for s in maison_scores) / len(maison_scores),
                "answer_relevancy":sum(s["answer_relevancy"] for s in maison_scores) / len(maison_scores),
                "context_recall":  sum(s["context_recall"]  for s in maison_scores) / len(maison_scores),
            },
            "ragas": ragas_scores,
        }
    }
    with open("ragas_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("💾 Résultats sauvegardés dans ragas_results.json")


if __name__ == "__main__":
    main()
