"""
Pipeline Q&A : recherche dans Pinecone + génération avec Claude
"""
import os
import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "ffessm-mft"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
N_RESULTS = 5

# Mots-clés dans la question → tag de niveau pour le metadata filtering
NIVEAU_KEYWORDS = {
    "niveau 1": "N1", "n1": "N1", "pe20": "N1",
    "niveau 2": "N2", "n2": "N2", "pa20": "N2", "pe40": "N2",
    "niveau 3": "N3", "n3": "N3", "pa40": "N3", "pe60": "N3", "pa60": "N3",
    "niveau 4": "N4", "n4": "N4", "guide de palanquée": "N4",
    "niveau 5": "N5", "n5": "N5",
    "initiateur": "Initiateur",
    "mf1": "MF1", "moniteur 1": "MF1",
    "mf2": "MF2", "moniteur 2": "MF2",
    "nitrox": "Nitrox", "trimix": "Trimix",
    "rifap": "RIFAP", "sidemount": "Sidemount",
}


def detect_niveau(question: str) -> str | None:
    """Détecte si la question mentionne un niveau spécifique."""
    q = question.lower()
    for keyword, tag in NIVEAU_KEYWORDS.items():
        if keyword in q:
            return tag
    return None

# Chargement du modèle une seule fois (évite de le recharger à chaque question)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    return f"""Tu es un assistant expert en plongée sous-marine et formations FFESSM.
Réponds à la question en te basant UNIQUEMENT sur les extraits du Manuel de Formation Technique (MFT) fournis ci-dessous.
Si la réponse n'est pas dans les extraits, dis-le clairement.

EXTRAITS MFT :
{context_block}

QUESTION : {question}

RÉPONSE :"""


def ask(question: str) -> tuple[str, list[str]]:
    # 1. Embedding de la question avec le même modèle qu'à l'ingestion
    model = get_model()
    question_vector = model.encode(question).tolist()

    # 2. Recherche dans Pinecone avec filtrage optionnel par niveau
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(INDEX_NAME)

    niveau = detect_niveau(question)
    query_kwargs = {
        "vector": question_vector,
        "top_k": N_RESULTS,
        "include_metadata": True,
    }
    if niveau:
        # Pinecone metadata filter : ne cherche que dans les chunks de ce niveau
        query_kwargs["filter"] = {"niveau": {"$eq": niveau}}

    results = index.query(**query_kwargs)

    # 3. Récupère le texte et les métadonnées depuis les résultats
    matches = results["matches"]
    contexts = [m["metadata"]["text"] for m in matches]
    sources = []
    seen = set()
    for m in matches:
        meta = m["metadata"]
        key = (meta["source"], meta.get("section", ""), meta.get("page", ""))
        if key not in seen:
            seen.add(key)
            label = meta["source"]
            if meta.get("section"):
                label += f" › {meta['section']}"
            if meta.get("page"):
                label += f" (p.{int(meta['page'])})"
            sources.append(label)

    # 4. Génération avec Claude
    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = claude.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": build_prompt(question, contexts)}],
    )

    return message.content[0].text, sources


def main():
    print("🤿 Chatbot FFESSM MFT — tape 'quit' pour quitter\n")
    while True:
        question = input("Question : ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("\n🔍 Recherche en cours...")
        answer, sources = ask(question)
        print(f"\n💬 {answer}")
        print("\n📚 Sources :")
        for s in sources:
            print(f"   • {s}")
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
