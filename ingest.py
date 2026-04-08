"""
Étape 3 & 4 : Chargement des PDFs + création des embeddings + stockage dans ChromaDB

Chunking intelligent :
- On extrait les blocs de texte page par page via PyMuPDF
- On détecte les titres de section (taille de police > seuil, ou texte tout en majuscules)
- On regroupe le texte par section → chaque chunk = une idée complète
- Si une section est trop longue, on la re-découpe avec overlap
"""
import fitz  # PyMuPDF
import re
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

DOCS_DIR = Path("docs")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "ffessm_mft"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MAX_CHUNK_WORDS = 400   # taille max d'un chunk (mots)
OVERLAP_WORDS = 50      # overlap si on re-découpe une section trop longue


def is_section_header(text: str, font_size: float, median_size: float) -> bool:
    """
    Détecte si un bloc de texte est un titre de section.
    Critères : police plus grande que la médiane, ou texte tout en majuscules courts.
    """
    text = text.strip()
    if not text or len(text) > 200:
        return False
    if font_size > median_size * 1.15:
        return True
    if text.isupper() and len(text.split()) <= 10:
        return True
    # Numérotation type "1.", "2.1", "A."
    if re.match(r"^(\d+\.)+\s+\w|^[A-Z]\.\s+\w", text):
        return True
    return False


def extract_sections(path: Path) -> list[dict]:
    """
    Extrait le texte d'un PDF découpé par sections détectées.
    Retourne une liste de dicts {"title": str, "content": str, "page": int}
    """
    doc = fitz.open(path)

    # Collecte tous les blocs avec leur taille de police
    all_blocks = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:  # 0 = texte
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        all_blocks.append({
                            "text": text,
                            "size": span["size"],
                            "page": page_num + 1,
                        })

    if not all_blocks:
        return []

    # Taille médiane pour détecter les titres relativement
    sizes = sorted(b["size"] for b in all_blocks)
    median_size = sizes[len(sizes) // 2]

    # Regroupe les blocs en sections
    sections = []
    current_title = path.stem  # titre par défaut = nom du fichier
    current_page = 1
    current_lines = []

    for block in all_blocks:
        if is_section_header(block["text"], block["size"], median_size):
            # On sauvegarde la section en cours avant d'en commencer une nouvelle
            if current_lines:
                sections.append({
                    "title": current_title,
                    "content": " ".join(current_lines),
                    "page": current_page,
                })
            current_title = block["text"]
            current_page = block["page"]
            current_lines = []
        else:
            current_lines.append(block["text"])

    # Dernière section
    if current_lines:
        sections.append({
            "title": current_title,
            "content": " ".join(current_lines),
            "page": current_page,
        })

    return sections


def split_if_too_long(section: dict, max_words: int, overlap: int) -> list[dict]:
    """
    Si le contenu d'une section dépasse max_words, on la re-découpe avec overlap.
    Chaque sous-chunk hérite du titre de section pour conserver le contexte.
    """
    words = section["content"].split()
    if len(words) <= max_words:
        return [section]

    chunks = []
    start = 0
    part = 0
    while start < len(words):
        end = start + max_words
        chunks.append({
            "title": section["title"],
            "content": " ".join(words[start:end]),
            "page": section["page"],
            "part": part,
        })
        start += max_words - overlap
        part += 1

    return chunks


def build_chunks(path: Path) -> tuple[list[str], list[dict]]:
    """
    Pipeline complet : extrait les sections, re-découpe si nécessaire.
    Retourne (textes, métadonnées) prêts pour ChromaDB.
    """
    sections = extract_sections(path)
    all_chunks = []
    for section in sections:
        all_chunks.extend(split_if_too_long(section, MAX_CHUNK_WORDS, OVERLAP_WORDS))

    # On préfixe chaque chunk avec son titre pour donner du contexte à l'embedding
    texts = [
        f"[{c['title']}]\n{c['content']}" for c in all_chunks
    ]
    metadatas = [
        {
            "source": path.name,
            "section": c["title"],
            "page": c["page"],
        }
        for c in all_chunks
    ]
    return texts, metadatas


def ingest():
    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print("❌ Aucun PDF trouvé dans le dossier docs/")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=ef)

    total_chunks = 0

    for pdf_path in pdfs:
        print(f"📄 Traitement : {pdf_path.name}")
        texts, metadatas = build_chunks(pdf_path)
        print(f"   → {len(texts)} chunks générés")

        if not texts:
            print("   → ignoré (vide)")
            continue

        ids = [f"{pdf_path.stem}_{i}" for i in range(len(texts))]
        collection.add(documents=texts, ids=ids, metadatas=metadatas)
        total_chunks += len(texts)

    print(f"\n✅ Ingestion terminée : {len(pdfs)} PDF(s), {total_chunks} chunks indexés")


if __name__ == "__main__":
    ingest()
