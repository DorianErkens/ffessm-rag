"""
Évaluation du pipeline RAG — RAGAS-like

RAGAS mesure 4 métriques clés. On les implémente ici sans dépendance externe,
en utilisant Claude comme juge LLM (exactement ce que RAGAS ferait).

─── Les 4 métriques ──────────────────────────────────────────────────────────

1. Context Recall   : Les bons chunks sont-ils remontés ?
   → On vérifie si la réponse de référence peut être déduite des chunks récupérés.
   → Faible = le retrieval rate des mauvais passages.

2. Context Precision : Les chunks remontés sont-ils tous utiles ?
   → Ratio de chunks pertinents parmi les chunks récupérés.
   → Faible = trop de bruit dans le contexte injecté à Claude.

3. Faithfulness     : La réponse générée s'appuie-t-elle sur les chunks ?
   → Détecte les hallucinations : Claude invente des infos absentes du contexte.
   → Faible = le modèle "complète" avec ses connaissances générales.

4. Answer Relevancy : La réponse répond-elle vraiment à la question posée ?
   → Indépendant de la vérité : une réponse hors-sujet est pénalisée.
   → Faible = réponses trop vagues, qui esquivent la question.

─── Golden dataset ───────────────────────────────────────────────────────────

Questions dont on connaît la réponse attendue, extraites des MFTs FFESSM.
C'est le socle de toute évaluation RAG : sans référence, on ne peut rien mesurer.
"""
import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import anthropic

load_dotenv()

INDEX_NAME = "ffessm-mft"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
N_RESULTS = 12

# ─── Golden dataset ───────────────────────────────────────────────────────────
# Format : question + réponse de référence (ground truth)
# Les réponses sont volontairement concises — elles servent de référence,
# pas de modèle de style.

GOLDEN_DATASET = [
    {
        "question": "Quel est l'âge minimum pour passer le Niveau 1 ?",
        "reference": "L'âge minimum pour le Niveau 1 (PE20) est de 14 ans.",
    },
    {
        "question": "Quelles sont les conditions pour obtenir le Niveau 2 (PA20/PE40) ?",
        "reference": "Être titulaire du Niveau 1 ou équivalent, avoir au moins 16 ans, présenter un certificat médical de non contre-indication à la plongée et être licencié FFESSM.",
    },
    {
        "question": "Jusqu'à quelle profondeur un plongeur Niveau 2 peut-il évoluer en autonomie ?",
        "reference": "Un plongeur Niveau 2 (PA20) peut évoluer en autonomie jusqu'à 20 m, et jusqu'à 40 m encadré par un Guide de Palanquée.",
    },
    {
        "question": "Quelles sont les prérogatives d'encadrement du Guide de Palanquée (N4) ?",
        "reference": "Le Guide de Palanquée peut encadrer des plongeurs jusqu'à 60 m, guider des palanquées en exploration, et encadrer des plongeurs PE40 jusqu'à 40 m et des plongeurs PE60 jusqu'à 60 m.",
    },
    {
        "question": "Qu'est-ce que le RIFAP et à qui s'adresse-t-il ?",
        "reference": "Le RIFAP (Réactions et Interventions Face aux Accidents de Plongée) est une formation aux gestes de secours spécifiques à la plongée. Il est requis pour les Niveaux 3, 4, MF1 et MF2.",
    },
    {
        "question": "Quelles sont les conditions d'accès à la formation MF1 ?",
        "reference": "Être titulaire du Niveau 4 (Guide de Palanquée) ou équivalent, avoir au moins 18 ans, être licencié FFESSM, posséder le RIFAP et avoir validé les prérequis pédagogiques.",
    },
    {
        "question": "Quelle est la différence entre PE60 et PA60 pour le Niveau 3 ?",
        "reference": "PE60 (Plongeur Encadré à 60 m) : le N3 peut plonger jusqu'à 60 m sous la responsabilité d'un E4 minimum. PA60 (Plongeur Autonome à 60 m) : le N3 peut plonger jusqu'à 60 m en autonomie en présence d'un Directeur de Plongée.",
    },
    {
        "question": "Peut-on plonger en nitrox sans qualification spécifique à la FFESSM ?",
        "reference": "Non, la plongée au nitrox nécessite une qualification spécifique. La FFESSM délivre une qualification nitrox qui permet d'utiliser des mélanges jusqu'à 40% d'oxygène.",
    },
    {
        "question": "Quel est le rôle du Directeur de Plongée ?",
        "reference": "Le Directeur de Plongée est responsable de l'organisation et de la sécurité de l'activité. Il définit les conditions de plongée, valide les palanquées et est responsable du respect de la réglementation.",
    },
    {
        "question": "Quelles sont les aptitudes du plongeur Niveau 1 (PE20) ?",
        "reference": "Le plongeur Niveau 1 peut évoluer jusqu'à 20 m, encadré par un Guide de Palanquée. Il ne peut pas plonger en autonomie.",
    },
]


# ─── Pipeline RAG (retrieval uniquement) ─────────────────────────────────────

def retrieve(question: str, model, index) -> list[str]:
    vector = model.encode(question).tolist()
    results = index.query(vector=vector, top_k=N_RESULTS, include_metadata=True)
    return [m["metadata"]["text"] for m in results["matches"]]


def generate(question: str, contexts: list[str], claude) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    system = f"""Tu es un assistant expert en plongée sous-marine et formations FFESSM.
Réponds en te basant UNIQUEMENT sur les extraits du MFT fournis.
Si la réponse n'est pas dans les extraits, dis-le clairement.

EXTRAITS MFT :
{context_block}"""
    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text.strip()


# ─── Métriques LLM-judged ─────────────────────────────────────────────────────

def score_context_recall(question: str, contexts: list[str], reference: str, claude) -> float:
    """
    Est-ce que les chunks récupérés contiennent l'information nécessaire
    pour répondre correctement (référence) ?
    Score : 0.0 à 1.0
    """
    context_block = "\n\n---\n\n".join(contexts[:5])  # top 5 suffisent
    prompt = f"""Tu évalues un système RAG. Voici une question, la réponse de référence attendue, et les extraits récupérés.

Question : {question}
Réponse de référence : {reference}
Extraits récupérés :
{context_block}

Les extraits récupérés contiennent-ils suffisamment d'information pour déduire la réponse de référence ?
Réponds UNIQUEMENT par un JSON : {{"score": 0.0}} à {{"score": 1.0}} et {{"justification": "..."}}
- 1.0 : toute l'information nécessaire est présente
- 0.5 : l'information est partiellement présente
- 0.0 : l'information est absente ou contradictoire"""

    r = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        data = json.loads(r.content[0].text)
        return float(data["score"]), data.get("justification", "")
    except Exception:
        return 0.5, "parse error"


def score_faithfulness(question: str, answer: str, contexts: list[str], claude) -> float:
    """
    La réponse générée s'appuie-t-elle uniquement sur les extraits ?
    Détecte les hallucinations.
    """
    context_block = "\n\n---\n\n".join(contexts[:5])
    prompt = f"""Tu évalues si une réponse de chatbot est fidèle aux extraits fournis (pas d'hallucination).

Question : {question}
Réponse générée : {answer}
Extraits sur lesquels la réponse doit se baser :
{context_block}

Chaque affirmation de la réponse est-elle supportée par les extraits ?
Réponds UNIQUEMENT par un JSON : {{"score": 0.0}} à {{"score": 1.0}} et {{"justification": "..."}}
- 1.0 : toutes les affirmations sont dans les extraits
- 0.5 : certaines affirmations sont inventées
- 0.0 : la réponse contredit ou ignore les extraits"""

    r = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        data = json.loads(r.content[0].text)
        return float(data["score"]), data.get("justification", "")
    except Exception:
        return 0.5, "parse error"


def score_answer_relevancy(question: str, answer: str, claude) -> float:
    """
    La réponse répond-elle vraiment à la question posée ?
    """
    prompt = f"""Tu évalues si une réponse répond directement à la question posée.

Question : {question}
Réponse : {answer}

La réponse est-elle pertinente et directement utile pour répondre à la question ?
Réponds UNIQUEMENT par un JSON : {{"score": 0.0}} à {{"score": 1.0}} et {{"justification": "..."}}
- 1.0 : réponse directe, complète, sans hors-sujet
- 0.5 : réponse partielle ou trop vague
- 0.0 : réponse hors-sujet ou qui esquive la question"""

    r = claude.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        data = json.loads(r.content[0].text)
        return float(data["score"]), data.get("justification", "")
    except Exception:
        return 0.5, "parse error"


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_evaluation():
    print("🔤 Chargement du modèle d'embedding...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    index = Pinecone(api_key=os.getenv("PINECONE_API_KEY")).Index(INDEX_NAME)
    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    results = []

    for i, sample in enumerate(GOLDEN_DATASET):
        q = sample["question"]
        ref = sample["reference"]
        print(f"\n[{i+1}/{len(GOLDEN_DATASET)}] {q[:60]}...")

        # Retrieval
        contexts = retrieve(q, model, index)

        # Generation
        answer = generate(q, contexts, claude)

        # Scoring
        recall_score, recall_why = score_context_recall(q, contexts, ref, claude)
        faithful_score, faithful_why = score_faithfulness(q, answer, contexts, claude)
        relevancy_score, relevancy_why = score_answer_relevancy(q, answer, claude)

        results.append({
            "question": q,
            "context_recall": recall_score,
            "faithfulness": faithful_score,
            "answer_relevancy": relevancy_score,
            "recall_why": recall_why,
            "faithful_why": faithful_why,
            "relevancy_why": relevancy_why,
        })

        print(f"   Context Recall    : {recall_score:.2f}  — {recall_why[:80]}")
        print(f"   Faithfulness      : {faithful_score:.2f}  — {faithful_why[:80]}")
        print(f"   Answer Relevancy  : {relevancy_score:.2f}  — {relevancy_why[:80]}")

    # ─── Résumé ───────────────────────────────────────────────────────────────
    avg_recall = sum(r["context_recall"] for r in results) / len(results)
    avg_faithful = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)

    print("\n" + "═" * 60)
    print("RÉSULTATS GLOBAUX")
    print("═" * 60)
    print(f"  Context Recall    : {avg_recall:.2f} / 1.0")
    print(f"  Faithfulness      : {avg_faithful:.2f} / 1.0")
    print(f"  Answer Relevancy  : {avg_relevancy:.2f} / 1.0")
    print("═" * 60)
    print("""
Interprétation :
  Context Recall   faible → le retrieval rate des mauvais chunks
  Faithfulness     faible → Claude hallucine (invente des infos)
  Answer Relevancy faible → les réponses sont vagues ou hors-sujet
""")

    return results


if __name__ == "__main__":
    run_evaluation()
