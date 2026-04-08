"""
Étape 3 & 4 : Chargement des PDFs + création des embeddings + stockage dans ChromaDB
"""
import os
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

DOCS_DIR = Path("docs")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "ffessm_mft"

# Modèle d'embedding léger, gratuit, local
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_pdf(path: Path) -> str:
    """Extrait le texte brut d'un PDF."""
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Découpe le texte en chunks avec overlap.

    chunk_size : nb de mots par chunk
    overlap    : nb de mots partagés entre deux chunks consécutifs
                 (évite de couper une info en deux)
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # on recule de `overlap` mots

    return chunks


def ingest():
    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print("❌ Aucun PDF trouvé dans le dossier docs/")
        print("   → Ajoute des fichiers MFT au format PDF dans ce dossier")
        return

    # Init ChromaDB (base vectorielle locale, stockée sur disque)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Recrée la collection proprement si elle existe déjà
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=ef)

    total_chunks = 0

    for pdf_path in pdfs:
        print(f"📄 Traitement : {pdf_path.name}")
        text = load_pdf(pdf_path)
        chunks = chunk_text(text)
        print(f"   → {len(chunks)} chunks générés")

        # Chaque chunk est stocké avec :
        # - un id unique
        # - le texte (document)
        # - des métadonnées (nom du fichier source)
        ids = [f"{pdf_path.stem}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf_path.name} for _ in chunks]

        if not chunks:
            print("   → ignoré (vide)")
            continue
        collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)

    print(f"\n✅ Ingestion terminée : {len(pdfs)} PDF(s), {total_chunks} chunks indexés")


if __name__ == "__main__":
    ingest()
