# FFESSM MFT — Assistant Formation

Chatbot de Q&A sur les Manuels de Formation Technique (MFT) de la FFESSM, construit avec un pipeline RAG (Retrieval-Augmented Generation).

**Interface web** : `streamlit run app.py`  
**Réindexer les docs** : `python ingest.py`

---

## Architecture

```
docs/*.pdf
    │
    ▼
[ ingest.py ] ─────────────────────────────────────────────────────
    │  1. Extraction du texte page par page (PyMuPDF)
    │  2. Détection des titres via taille de police → chunking par section
    │  3. Re-découpage avec overlap si section > 400 mots
    │  4. Préfixage du titre dans le texte → meilleur signal sémantique
    │  5. Génération des embeddings (paraphrase-multilingual-MiniLM-L12-v2)
    │  6. Stockage dans Pinecone avec métadonnées : source, section, page, niveau
    ▼
[ Pinecone (cloud) ] ← 4133 chunks indexés, accessibles depuis n'importe où
    │
    │   À chaque question :
    │
    ▼
[ app.py / chat.py ] ──────────────────────────────────────────────
    │  1. Détection du niveau mentionné dans la question (N1, N2, MF1…)
    │  2. Embedding de la question avec le même modèle
    │  3. Recherche des 5 chunks les plus proches (similarité cosinus)
    │     → filtrage metadata Pinecone si niveau détecté
    │  4. Injection des chunks dans le prompt Claude
    │  5. Génération en streaming
    ▼
[ Réponse + sources (fichier › section › page) ]
```

---

## Stack

| Composant | Outil | Pourquoi |
|-----------|-------|----------|
| Extraction PDF | PyMuPDF (`fitz`) | Accès aux métadonnées typographiques (taille police) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | Modèle multilingue, français natif |
| Base vectorielle | Pinecone (cloud) | Persist entre déploiements, accessible depuis Streamlit Cloud |
| LLM | Claude Opus 4.6 (Anthropic API) | Génération + streaming |
| Interface web | Streamlit | Déploiement zero-infra |

---

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
PINECONE_API_KEY=pcsk_...
```

---

## Usage

### Interface web
```bash
streamlit run app.py
```

### Ligne de commande
```bash
python chat.py
```

### Réindexer les documents
Placer les PDFs MFT dans `docs/`, puis :
```bash
python ingest.py
```

> L'ingestion est à faire une seule fois (ou quand les docs changent).  
> L'index Pinecone persiste dans le cloud.

---

## Déploiement Streamlit Cloud

1. Push le repo sur GitHub
2. Aller sur [share.streamlit.io](https://share.streamlit.io)
3. Connecter le repo, fichier principal : `app.py`
4. Ajouter les secrets dans **Settings > Secrets** :
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
PINECONE_API_KEY = "pcsk_..."
```

---

## Concepts clés

**RAG (Retrieval-Augmented Generation)** : au lieu de fine-tuner un modèle sur les MFTs (coûteux), on pré-indexe les documents sous forme de vecteurs. À chaque question, on récupère les passages les plus pertinents et on les injecte dans le prompt — le LLM répond en s'appuyant sur ces extraits.

**Embeddings** : représentation numérique du sens d'un texte. Deux phrases sémantiquement proches ont des vecteurs proches dans l'espace. C'est ce qui permet la recherche par sens, pas par mot-clé.

**Chunking sémantique** : on ne découpe pas les PDFs à l'aveugle par nombre de mots. PyMuPDF donne accès à la taille de police de chaque bloc — on s'en sert pour détecter les titres de section et couper aux frontières des idées.

**Metadata filtering** : chaque chunk stocké dans Pinecone porte un tag `niveau` (N1, N2, MF1…). Quand une question mentionne un niveau, on filtre en amont de la recherche vectorielle — résultats bien plus précis.
