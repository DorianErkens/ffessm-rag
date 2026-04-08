# FFESSM MFT RAG Chatbot

Chatbot de Q&A sur les Manuels de Formation Technique (MFT) de la FFESSM, construit avec un pipeline RAG (Retrieval-Augmented Generation).

## Architecture

```
docs/*.pdf
    │
    ▼
[ ingest.py ]
    │  1. Extraction du texte (PyMuPDF)
    │  2. Découpage en chunks
    │  3. Création des embeddings (all-MiniLM-L6-v2, local)
    │  4. Stockage dans ChromaDB (local, sur disque)
    ▼
[ chroma_db/ ]  ← base vectorielle persistante
    │
    │   Au moment d'une question :
    │
    ▼
[ chat.py ]
    │  1. Embedding de la question
    │  2. Recherche des 5 chunks les plus proches (similarité cosinus)
    │  3. Injection des chunks dans le prompt
    │  4. Génération de la réponse via Claude (Anthropic API)
    ▼
[ Réponse + sources ]
```

## Stack

| Composant | Outil |
|-----------|-------|
| Extraction PDF | PyMuPDF (`fitz`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, gratuit) |
| Base vectorielle | ChromaDB (local) |
| LLM | Claude claude-opus-4-6 (Anthropic API) |

## Installation

```bash
git clone https://github.com/DorianErkens/ffessm-rag
cd ffessm-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Créer un fichier `.env` :
```
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### 1. Indexer les documents

Placer les PDFs MFT dans le dossier `docs/`, puis :

```bash
python ingest.py
```

### 2. Lancer le chatbot

```bash
python chat.py
```

## Concepts clés

**RAG (Retrieval-Augmented Generation)** : au lieu de fine-tuner un LLM sur les MFTs (coûteux), on pré-indexe les documents sous forme de vecteurs. À chaque question, on récupère les passages les plus pertinents et on les injecte dans le contexte du LLM — qui génère alors une réponse ancrée dans les sources.

**Embeddings** : représentation numérique du sens d'un texte. Deux phrases sémantiquement proches ont des vecteurs proches dans l'espace. C'est ce qui permet la recherche par sens et non par mot-clé.

**Chunking** : les PDFs sont découpés en segments pour que chaque morceau indexé soit assez petit pour être précis, mais assez grand pour être cohérent.
