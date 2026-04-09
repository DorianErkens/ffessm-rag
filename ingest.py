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
import unicodedata
import fitz  # PyMuPDF
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

DOCS_DIR = Path("docs")
INDEX_NAME = "ffessm-mft"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"  # modèle multilingue, 768 dims, meilleur que MiniLM
EMBEDDING_DIM = 768   # dimension des vecteurs produits par ce modèle
BATCH_SIZE = 100      # nb de vecteurs envoyés à Pinecone en une seule requête

MAX_CHUNK_WORDS = 400
OVERLAP_WORDS = 80   # ~20% d'overlap pour ne pas couper les infos de transition


# ─── Nettoyage du texte extrait ───────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Les PDFs FFESSM utilisent une typo espacée : 'N I V E A U  1', 'P L O N G E U R  N I V E A U  1'.
    - Espace simple  = séparation entre lettres d'un même mot
    - Double espace  = séparation entre mots

    Algorithme :
    1. On découpe sur les doubles espaces → groupes de lettres
    2. Si toutes les parties d'un groupe font ≤ 2 chars → c'est de la typo espacée → on recolle
    3. On recolle les groupes avec un espace simple

    'N I V E A U  1'           → ['N I V E A U', '1']     → 'NIVEAU 1'
    'P L O N G E U R  N I V E A U' → ['PLONGEUR', 'NIVEAU'] → 'PLONGEUR NIVEAU'
    'D É C E M B R E  2 0 2 5' → ['DÉCEMBRE', '2025']    → 'DÉCEMBRE 2025'
    """
    parts = re.split(r" {2,}", text)
    cleaned_parts = []
    for part in parts:
        tokens = part.split(" ")
        if len(tokens) > 2 and all(len(t) <= 2 for t in tokens if t):
            cleaned_parts.append("".join(tokens))
        else:
            cleaned_parts.append(part)

    result = " ".join(cleaned_parts)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ─── Extraction du niveau depuis le nom de fichier ────────────────────────────

# Mapping explicite : nom de fichier (partiel) → tag de niveau
# Permet le metadata filtering dans Pinecone ("donne-moi seulement les chunks N2")
NIVEAU_MAP = {
    "Niveau 1": "N1", "PE20": "N1",
    "Niveau 2": "N2", "PA20": "N2", "PE40": "N2",
    "Niveau 3": "N3", "PA40": "N3", "PE60": "N3", "PA60": "N3",
    "Niveau 4": "N4", "Guide de palanquée": "N4",
    "Niveau 5": "N5", "Directeur de plongée": "N5",
    "Initiateur": "Initiateur",
    "MF1": "MF1", "Moniteur Fédéral 1": "MF1",
    "MF2": "MF2", "Moniteur Fédéral 2": "MF2",
    "BPJEPS": "BPJEPS", "DEJEPS": "DEJEPS", "BEPPA": "BEPPA",
    "nitrox": "Nitrox", "trimix": "Trimix",
    "recycleur circuit fermé": "Recycleur-CCR",
    "recycleur circuit semi-fermé": "Recycleur-SCR",
    "Sidemount": "Sidemount", "RIFAP": "RIFAP",
    "PA12": "PA12", "PE12": "PE12",
    "Jeunes": "Jeunes", "HANDISUB": "Handisub",
    "Randonnée": "Randonnée-sub",
}


def extract_niveau(filename: str) -> str:
    """Retourne le tag de niveau/catégorie à partir du nom de fichier."""
    for keyword, tag in NIVEAU_MAP.items():
        if keyword.lower() in filename.lower():
            return tag
    return "Général"


# ─── Chunking ────────────────────────────────────────────────────────────────

def is_section_header(text: str, font_size: float, median_size: float) -> bool:
    text = text.strip()
    # Trop court ou trop long → pas un titre
    if len(text) < 4 or len(text) > 200:
        return False
    # Décoratif (tirets, symboles, chiffres seuls)
    if re.match(r"^[―—\-–•·\d]+$", text):
        return False
    # Codes de certification FFESSM (N1, N2, PE20, PA40, MF1, MF2, BPJEPS...)
    # Ces badges décoratifs ne sont pas des titres de section
    if re.match(r"^(N\d|PE\d+|PA\d+|MF\d|E\d|GP|DP|BPJEPS|DEJEPS|BEPPA|RIFAP|TIV|ANTEOR)$", text):
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
                    text = clean_text(span["text"].strip())
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


MIN_CHUNK_WORDS = 30  # on ignore les chunks trop courts (titres seuls, résidus)


def build_chunks(path: Path) -> tuple[list[str], list[dict]]:
    """Retourne (textes à embedder, métadonnées associées)."""
    all_chunks = []
    for section in extract_sections(path):
        for chunk in split_if_too_long(section):
            # Filtre les chunks sans contenu réel
            if len(chunk["content"].split()) >= MIN_CHUNK_WORDS:
                all_chunks.append(chunk)

    # On préfixe le titre au contenu : améliore la qualité de l'embedding
    texts = [f"[{c['title']}]\n{c['content']}" for c in all_chunks]
    niveau = extract_niveau(path.name)
    metadatas = [
        {
            "text": texts[i],
            "source": path.name,
            "section": c["title"],
            "page": c["page"],
            "niveau": niveau,   # permet le filtrage metadata dans Pinecone
        }
        for i, c in enumerate(all_chunks)
    ]
    return texts, metadatas


# ─── Pinecone ─────────────────────────────────────────────────────────────────

def get_or_create_index(pc: Pinecone) -> object:
    """
    Crée l'index Pinecone s'il n'existe pas, puis vide tous les vecteurs existants.
    Si la dimension de l'index existant ne correspond pas (changement de modèle),
    supprime et recrée l'index avec la bonne dimension.
    """
    existing = {idx.name: idx for idx in pc.list_indexes()}
    if INDEX_NAME not in existing:
        print(f"🔧 Création de l'index Pinecone '{INDEX_NAME}' ({EMBEDDING_DIM} dims)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("   → Index créé")
    else:
        current_dim = existing[INDEX_NAME].dimension
        if current_dim != EMBEDDING_DIM:
            print(f"⚠️  Dimension mismatch ({current_dim} → {EMBEDDING_DIM}), suppression et recréation...")
            pc.delete_index(INDEX_NAME)
            import time; time.sleep(10)  # attendre la suppression effective
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            print("   → Index recréé")
        else:
            print(f"🧹 Nettoyage de l'index existant '{INDEX_NAME}'...")
            index = pc.Index(INDEX_NAME)
            index.delete(delete_all=True)
            print("   → Index vidé")
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

        # ID unique par chunk — Pinecone exige de l'ASCII pur
        safe_stem = unicodedata.normalize("NFD", pdf_path.stem)
        safe_stem = safe_stem.encode("ascii", "ignore").decode("ascii")
        safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", safe_stem)
        ids = [f"{safe_stem}_{i}" for i in range(len(texts))]

        upsert_in_batches(index, ids, vectors, metadatas)
        print(f"   → {len(texts)} chunks indexés")
        total_chunks += len(texts)

    print(f"\n✅ Ingestion terminée : {len(pdfs)} PDF(s), {total_chunks} chunks dans Pinecone")


if __name__ == "__main__":
    ingest()
