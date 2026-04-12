# FFESSM MFT — Assistant Formation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ffessm-rag-proto.streamlit.app)

Deux modules complémentaires autour des Manuels de Formation Technique (MFT) de la FFESSM :

- **Assistant MFT** — chatbot RAG pour répondre aux questions réglementaires et factuelles
- **Générateur d'Éducatifs** — pipeline multi-agents LangGraph pour aider les moniteurs à préparer leurs séances

**Interface web** : `streamlit run app.py`  
**Réindexer les docs** : `python ingest.py`  
**Tests** : `pytest tests/`  
**Évaluation RAG** : `python evaluate.py`  
**LangGraph Studio** : `langgraph dev` → https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

---

## Architecture

### Module 1 — Assistant RAG

```
docs/*.pdf
    │
    ▼
[ ingest.py ] ─────────────────────────────────────────────────────────────
    │  1. Extraction du texte page par page (PyMuPDF)
    │  2. Nettoyage de la typo espacée ("N I V E A U" → "NIVEAU")
    │  3. Détection des titres de section via taille de police
    │  4. Chunking sémantique par section (overlap 80 mots)
    │  5. Filtrage des chunks < 30 mots (artefacts de mise en page)
    │  6. Génération des embeddings (paraphrase-multilingual-MiniLM-L12-v2)
    │  7. Nettoyage de l'index Pinecone puis upload des vecteurs
    ▼
[ Pinecone (cloud) ] ← ~1200 chunks propres, accessibles depuis n'importe où
    │
    │   À chaque question :
    │
    ▼
[ app.py ] ─────────────────────────────────────────────────────────────────
    │  1. Détection du niveau dans la question (N1, N2, MF1…)
    │  2. Embedding de la question
    │  3. Recherche des 8 chunks les plus proches (similarité cosinus)
    │     → filtrage metadata Pinecone si niveau détecté
    │  4. Injection des chunks dans le prompt Claude
    │  5. Génération en streaming
    ▼
[ Réponse + sources (fichier › section › page) ]
```

### Module 2 — Générateur d'Éducatifs (LangGraph)

Pipeline multi-agents déclenché quand un moniteur pose une question pédagogique.

```
[ Question moniteur ]
        │
        ▼
[ Node 1 — Intent Classifier ]   Claude Haiku
  Détecte si la question est pédagogique ("enseigner")
  ou factuelle ("info") → bifurcation du graphe
        │
        │ si "enseigner"
        ▼
[ Node 2 — Competency Extractor ]   Claude Haiku
  Extrait : compétence, niveau cible, stade de l'élève,
  type de séance (initiation / remédiation / perfectionnement)
        │
        ▼
[ Node 3 — RAG Retriever ]   Pinecone
  Construit une requête pédagogique optimisée
  et récupère les extraits MFT pertinents
        │
        ▼
[ Node 4 — Educatif Generator ]   Claude Sonnet + tool_use
  Génère la fiche éducatif structurée via function calling
  (JSON garanti valide par l'API Anthropic)
        │
        ▼
[ Fiche : contexte · objectif · position formation · justification
          prérequis · éducatifs progressifs · évaluation · pour aller plus loin ]
```

![Graphe LangGraph](docs/agent_graph.png)

---

## Stack

| Composant | Outil | Pourquoi |
|-----------|-------|----------|
| Extraction PDF | PyMuPDF (`fitz`) | Accès aux métadonnées typographiques (taille police) |
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` | Multilingue, français natif, gratuit |
| Base vectorielle | Pinecone (cloud) | Persiste entre déploiements, accessible depuis Streamlit Cloud |
| LLM génération | Claude Sonnet 4.6 (Anthropic API) | Génération + streaming + tool_use |
| LLM classification | Claude Haiku 4.5 | Tâches légères : intent, extraction d'entités |
| Orchestration agents | LangGraph | Graphe d'états typé, stream() natif, LangGraph Studio |
| Interface web | Streamlit multi-pages | Déploiement zero-infra |
| Observabilité | Langsmith | Traces LLM, évaluation feedback |

---

## Installation

```bash
git clone https://github.com/DorianErkens/ffessm-rag
cd ffessm-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -U "langgraph-cli[inmem]"   # pour LangGraph Studio
```

Créer un fichier `.env` :
```
ANTHROPIC_API_KEY=sk-ant-...
PINECONE_API_KEY=pcsk_...
LANGSMITH_API_KEY=ls__...   # optionnel — observabilité Langsmith
```

---

## Usage

### Interface web complète (RAG + Éducatifs)
```bash
streamlit run app.py
```

### Réindexer les documents
Placer les PDFs MFT dans `docs/`, puis :
```bash
python ingest.py
```

> L'ingestion vide l'index Pinecone et repart de zéro à chaque exécution.

### Tester le pipeline éducatifs en ligne de commande
```bash
python agents/graph.py
```

### LangGraph Studio (visualisation interactive du graphe)
```bash
langgraph dev
```
Puis ouvrir : https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

### Lancer les tests
```bash
pytest tests/ -v
```

---

## Déploiement Streamlit Cloud

1. Push le repo sur GitHub
2. Aller sur [share.streamlit.io](https://share.streamlit.io)
3. Connecter le repo, fichier principal : `app.py`
4. **Manage app** (en bas à droite) → **Settings** → **Secrets**
5. Coller les secrets au format TOML — **chaque clé doit tenir sur une seule ligne, sans retour à la ligne dans la valeur** :

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
PINECONE_API_KEY = "pcsk_..."
```

> **Piège fréquent** : si vous copiez-collez depuis un terminal ou un PDF, des retours à la ligne peuvent s'insérer au milieu de la clé → erreur `Invalid format: please enter valid TOML`. Vérifiez que chaque ligne est continue du `=` jusqu'au guillemet fermant.

6. **Save** → l'app redémarre automatiquement (~1 min)
7. Premier démarrage : ~3 min (téléchargement du modèle `sentence-transformers`)

---

## Évaluation du pipeline (RAGAS-like)

`evaluate.py` mesure la qualité du pipeline RAG sur 10 questions de référence, sans dépendance externe — Claude Haiku est utilisé comme juge LLM.

### Les 3 métriques

| Métrique | Ce qu'elle mesure | Diagnostic si faible |
|---|---|---|
| **Context Recall** | Les bons chunks sont-ils remontés ? | Le retrieval rate des mauvais passages |
| **Faithfulness** | La réponse s'appuie-t-elle sur les chunks ? | Claude hallucine (invente des infos) |
| **Answer Relevancy** | La réponse répond-elle vraiment à la question ? | Réponses vagues ou hors-sujet |

### Évolution des scores

| Run | Config | Context Recall | Faithfulness | Answer Relevancy |
|---|---|---|---|---|
| 1 | 8 chunks, filtre `$eq` | 0.53 | 0.62 | 0.70 |
| 2 | 12 chunks, filtre `$in` + Général | 0.59 | 0.45 | 0.78 |
| 3 | 8 chunks, filtre `$in` + Général + **eval fixée** | **0.70** | **0.72** | **0.74** |

> Run 1→3 : la majorité du gain vient de la correction du bug dans `evaluate.py` (retrieve sans filtre), pas d'un changement du pipeline applicatif. Les runs 1 et 2 mesuraient partiellement une fausse version du pipeline.

### Décisions de design issues de l'évaluation

**N_RESULTS = 8 (et non 12)** : passer à 12 chunks améliorait le Context Recall (+0.06) mais dégradait la Faithfulness (-0.17) en injectant trop de bruit dans le contexte Claude.

**Filtre `$in [niveau, "Général"]`** : quand un niveau est détecté dans la question, on inclut systématiquement les docs "Général" (règles fédérales transversales — Code du Sport, généralités brevets). Sans ça, une question N3 rate les contraintes légales générales.

**evaluate.py doit reproduire exactement app.py** : bug découvert — l'ancienne version de `retrieve()` dans `evaluate.py` ne filtrait pas. Les chunks RIFAP (score vectoriel 0.75) n'apparaissaient pas car battus par du bruit non filtré (0.47). Les scores RIFAP étaient faussement à 0.00.

---

## Historique des bugs — spécificités des PDFs FFESSM

Les MFTs FFESSM ont une mise en page complexe qui a nécessité plusieurs corrections.
Ces bugs sont couverts par les tests dans `tests/test_ingest.py`.

### Bug #1 — Typographie espacée
**Symptôme** : Chunks contenant `N I V E A U  1` au lieu de `NIVEAU 1`. Embeddings dégradés, recherches qui ratent.  
**Cause** : Les titres décoratifs des PDFs espacent chaque lettre. PyMuPDF extrait le texte tel quel.  
**Fix** : `clean_text()` détecte les séquences de lettres isolées séparées par des espaces simples et les recolle. Les doubles espaces marquent les frontières de mots.

### Bug #2 — Caractères isolés détectés comme titres de section
**Symptôme** : Chunks avec titre `[N]` et contenu commençant par `4 ...`.  
**Cause** : Dans le PDF, "N" et "4" apparaissent comme spans typographiques séparés. "N" a une police large → détecté comme titre.  
**Fix** : `is_section_header()` exige `len(text) >= 4`.

### Bug #3 — Codes de certification détectés comme titres
**Symptôme** : Chunks `[PE40]`, `[MF1]`, `[PA20]` avec contenu `)` — fragments inutilisables.  
**Cause** : Ces badges décoratifs (tout en majuscules, courts) passaient le filtre "isupper + <= 10 mots".  
**Fix** : Regex d'exclusion explicite des codes FFESSM dans `is_section_header()`.

### Bug #4 — Symboles décoratifs détectés comme titres
**Symptôme** : Sections découpées à chaque `―` (tiret long décoratif).  
**Fix** : Exclusion des textes composés uniquement de symboles `[―—\-–•·\d]`.

### Bug #5 — Chunks vides indexés et remontés en recherche
**Symptôme** : Claude répondait "les extraits ne contiennent aucune information exploitable". Les chunks récupérés étaient des titres seuls ou des cellules de tableau d'un mot.  
**Cause** : Toutes les sections étaient indexées, y compris les micro-fragments.  
**Fix** : Filtre `MIN_CHUNK_WORDS = 30` — les chunks dont le contenu fait moins de 30 mots sont ignorés.

### Bug #6 — IDs non-ASCII rejetés par Pinecone
**Symptôme** : `PineconeApiException: Vector ID must be ASCII` sur les PDFs avec accents dans le nom.  
**Cause** : Les IDs Pinecone doivent être en ASCII pur. Nos IDs étaient construits depuis les noms de fichiers (`vêtement étanche_0`).  
**Fix** : `unicodedata.normalize("NFD")` + `encode("ascii", "ignore")` + `re.sub` des caractères restants.

### Bug #7 — Chunks obsolètes persistant dans Pinecone
**Symptôme** : Après correction du chunking, les anciens chunks bugués revenaient encore dans les résultats.  
**Cause** : `index.upsert()` ne supprime pas les vecteurs existants — il ajoute ou met à jour par ID. Les anciens chunks avec des IDs différents restaient en base indéfiniment.  
**Fix** : `index.delete(delete_all=True)` au début de chaque ingestion pour repartir d'un index vide.

---

## Structure du projet

```
ffessm-rag/
├── app.py                    # Assistant RAG — interface Streamlit principale
├── ingest.py                 # Pipeline d'ingestion PDF → Pinecone
├── evaluate.py               # Évaluation RAG (context recall, faithfulness, relevancy)
├── agents/
│   ├── state.py              # TypedDict du state LangGraph
│   ├── intent_classifier.py  # Node 1 — classification intention (Haiku)
│   ├── competency_extractor.py # Node 2 — extraction compétence/niveau (Haiku)
│   ├── rag_retriever.py      # Node 3 — retrieval Pinecone pédagogique
│   ├── educatif_generator.py # Node 4 — génération fiche via tool_use (Sonnet)
│   └── graph.py              # Assembly LangGraph + point d'entrée public
├── pages/
│   └── 1_Éducatifs.py       # Page Streamlit module éducatifs
├── data/
│   └── competences_mft.json  # Référentiel compétences MFT (à enrichir)
├── docs/
│   ├── *.pdf                 # MFTs FFESSM source
│   └── agent_graph.png       # Visualisation du graphe LangGraph
├── tests/
│   └── test_ingest.py        # Tests unitaires pipeline d'ingestion
└── langgraph.json            # Config LangGraph Studio
```

---

## Concepts clés

**RAG (Retrieval-Augmented Generation)** : au lieu de fine-tuner un modèle sur les MFTs (coûteux), on pré-indexe les documents sous forme de vecteurs. À chaque question, on récupère les passages les plus pertinents et on les injecte dans le prompt — le LLM répond en s'appuyant sur ces extraits.

**Embeddings** : représentation numérique du sens d'un texte. Deux phrases sémantiquement proches ont des vecteurs proches dans l'espace. C'est ce qui permet la recherche par sens, pas par mot-clé.

**Chunking sémantique** : on ne découpe pas les PDFs à l'aveugle par nombre de mots. PyMuPDF donne accès à la taille de police de chaque bloc — on s'en sert pour détecter les titres de section et couper aux frontières des idées.

**Metadata filtering** : chaque chunk stocké dans Pinecone porte un tag `niveau` (N1, N2, MF1…). Quand une question mentionne un niveau, on filtre en amont de la recherche vectorielle — résultats bien plus précis.

**LangGraph** : framework d'orchestration d'agents basé sur un graphe d'états typés. Chaque nœud est une fonction pure `(state) → dict` qui retourne uniquement les clés qu'il modifie. Le graphe gère la fusion des states, les arêtes conditionnelles et l'observabilité native via `stream()`.

**Tool Use (Structured Output)** : au lieu de demander à Claude de "retourner un JSON" (fragile), on définit un schéma de fonction et on force Claude à l'appeler via `tool_choice`. L'API Anthropic garantit que le résultat est un dict Python valide respectant le schéma — zéro parsing, zéro guillemets non échappés.
