"""
Étape 5 : Pipeline Q&A — recherche + génération avec Claude
"""
import os
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "ffessm_mft"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_RESULTS = 5  # nb de chunks récupérés par question


def build_prompt(question: str, contexts: list[str]) -> str:
    """Construit le prompt avec les passages pertinents injectés."""
    context_block = "\n\n---\n\n".join(contexts)
    return f"""Tu es un assistant expert en plongée sous-marine et formations FFESSM.
Réponds à la question en te basant UNIQUEMENT sur les extraits du Manuel de Formation Technique (MFT) fournis ci-dessous.
Si la réponse n'est pas dans les extraits, dis-le clairement.

EXTRAITS MFT :
{context_block}

QUESTION : {question}

RÉPONSE :"""


def ask(question: str) -> str:
    # 1. Connexion à ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)

    # 2. Recherche sémantique : trouve les N chunks les plus proches
    results = collection.query(query_texts=[question], n_results=N_RESULTS)
    contexts = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    # 3. Génération avec Claude
    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    message = claude.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": build_prompt(question, contexts)}],
    )

    answer = message.content[0].text
    # Formate les sources avec section + page si dispo
    seen = set()
    unique_sources = []
    for m in results["metadatas"][0]:
        key = (m["source"], m.get("section", ""), m.get("page", ""))
        if key not in seen:
            seen.add(key)
            label = m["source"]
            if m.get("section"):
                label += f" › {m['section']}"
            if m.get("page"):
                label += f" (p.{m['page']})"
            unique_sources.append(label)

    return answer, unique_sources


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
        print()


if __name__ == "__main__":
    main()
