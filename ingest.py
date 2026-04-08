"""
Ingestion des PDFs MFT → embeddings → Pinecone

Chunking intelligent :
- On extrait les blocs de texte page par page via PyMuPDF
- On détecte les titres de section (taille de police > seuil, ou texte tout en majuscules)
- On regroupe le texte par section → chaque chunk = une idée complète
- Si une section est trop longue, on la re-découpe avec overlap

Pourquoi Pinecone plutôt que ChromaDB local ?
- La base vectorielle vit dans le cloud → accessible depuis n'importe quel serveur
- On ingère une seule fois, l'app web (Streamlit) requête Pinecone directement
"""
import os
import re
import fitz  # PyMuPDF
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

DOCS_DIR = Path("docs")
INDEX_NAME = "ffessm-mft"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # modèle multilingue (fr/en)
EMBEDDING_DIM = 384   # dimension des vecteurs produits par ce modèle
BATCH_SIZE = 100      # nb de vecteurs envoyés à Pinecone en une seule requête

MAX_CHUNK_WORDS = 400
OVERLAP_WORDS = 50


# ─── Chunking ────────────────────────────────────────────────────────────────

def is_section_header(text: str, font_size: float, median_size: float) -> bool:
    text = text.strip()
    if not text or len(text) > 200:
        return False
    if font_size > median_size * 1.15:
        return True
    if text.isupper() and len(text.split()) <= 10:
        return True
    if re.match(r"^(\d+\.)+\s+\w|^[A-Z]\.\s+\w", text):
        return True
    return False


def extract_sections(path: Path) -> list[dict]:
    doc = fitz.open(path)
    all_blocks = []

    for page_num, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
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

    sizes = sorted(b["size"] for b in all_blocks)
    median_size = sizes[len(sizes) // 2]

    sections = []
    current_title = path.stem
    current_page = 1
    current_lines = []

    for block in all_blocks:
        if is_section_header(block["text"], block["size"], median_size):
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

    if current_lines:
        sections.append({
            "title": current_title,
            "content": " ".join(current_lines),
            "page": current_page,
        })

    return sections


def split_if_too_long(section: dict) -> list[dict]:
    words = section["content"].split()
    if len(words) <= MAX_CHUNK_WORDS:
        return [section]

    chunks, start, part = [], 0, 0
    while start < len(words):
        chunks.append({
            "title": section["title"],
            "content": " ".join(words[start:start + MAX_CHUNK_WORDS]),
            "page": section["page"],
            "part": part,
        })
        start += MAX_CHUNK_WORDS - OVERLAP_WORDS
        part += 1
    return chunks


def build_chunks(path: Path) -> tuple[list[str], list[dict]]:
    """Retourne (textes à embedder, métadonnées associées)."""
    all_chunks = []
    for section in extract_sections(path):
        all_chunks.extend(split_if_too_long(section))

    # On préfixe le titre au contenu : améliore la qualité de l'embedding
    texts = [f"[{c['title']}]\n{c['content']}" for c in all_chunks]
    metadatas = [
        {
            "text": texts[i],       # on stocke le texte brut dans les métadonnées
            "source": path.name,    # pour l'affichage des sources
            "section": c["title"],
            "page": c["page"],
        }
        for i, c in enumerate(all_chunks)
    ]
    return texts, metadatas


# ─── Pinecone ─────────────────────────────────────────────────────────────────

def get_or_create_index(pc: Pinecone) -> object:
    """
    Crée l'index Pinecone s'il n'existe pas encore.
    Un index = une base vectorielle. On précise :
    - dimension : taille des vecteurs (dépend du modèle d'embedding)
    - metric    : comment on mesure la similarité (cosine = standard pour le texte)
    """
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"🔧 Création de l'index Pinecone '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("   → Index créé")
    return pc.Index(INDEX_NAME)


def upsert_in_batches(index, ids, vectors, metadatas):
    """
    Pinecone recommande d'envoyer les vecteurs par batch (pas tout d'un coup).
    Chaque vecteur = un dict avec id, values (le vecteur), metadata.
    """
    for i in range(0, len(ids), BATCH_SIZE):
        batch = [
            {"id": ids[j], "values": vectors[j].tolist(), "metadata": metadatas[j]}
            for j in range(i, min(i + BATCH_SIZE, len(ids)))
        ]
        index.upsert(vectors=batch)


# ─── Pipeline principal ───────────────────────────────────────────────────────

def ingest():
    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print("❌ Aucun PDF trouvé dans docs/")
        return

    print(f"🔤 Chargement du modèle d'embedding ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = get_or_create_index(pc)

    total_chunks = 0

    for pdf_path in pdfs:
        print(f"📄 {pdf_path.name}")
        texts, metadatas = build_chunks(pdf_path)

        if not texts:
            print("   → ignoré (vide)")
            continue

        # Génère les embeddings pour tous les chunks du PDF
        vectors = model.encode(texts, show_progress_bar=False)

        # ID unique par chunk : nom_du_fichier_index
        ids = [f"{pdf_path.stem}_{i}" for i in range(len(texts))]

        upsert_in_batches(index, ids, vectors, metadatas)
        print(f"   → {len(texts)} chunks indexés")
        total_chunks += len(texts)

    print(f"\n✅ Ingestion terminée : {len(pdfs)} PDF(s), {total_chunks} chunks dans Pinecone")


if __name__ == "__main__":
    ingest()
