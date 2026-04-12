"""
Node 3 — RAG Retriever

Rôle : interroger Pinecone pour récupérer les extraits MFT pertinents
sur la compétence ciblée, filtrés par niveau si disponible.

Pourquoi ce nœud existe alors que app.py fait déjà du RAG ?
Le RAG de app.py est conçu pour répondre à des questions factuelles.
Ici, on cherche des extraits sur UNE COMPÉTENCE SPÉCIFIQUE pour alimenter
le générateur d'éducatifs — la requête est construite différemment :
elle combine la compétence ET le contexte pédagogique (type de séance).

Concept clé : Query construction
La qualité du retrieval dépend à 80% de la qualité de la requête embedée.
On ne passe pas la question brute du moniteur — on construit une requête
optimisée pour trouver les bons chunks MFT sur la compétence.

Ex : question = "Mon élève N1 a du mal à palmer sans remonter"
     requête construite = "palmation technique éducatif N1 erreurs fréquentes"
     → cherche des chunks qui parlent de palmation au niveau N1, pas des conditions d'accès
"""
import os

try:
    from agents.state import EducatifState
except ImportError:
    from state import EducatifState  # type: ignore

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


INDEX_NAME = "ffessm-mft"
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"
N_RESULTS = 6  # moins que le RAG classique (8) : on veut de la précision, pas du recall


# ─── Cache module-level ───────────────────────────────────────────────────────
# Dans Streamlit, on utilise @st.cache_resource. Ici (module Python pur),
# on cache simplement au niveau module : les objets sont créés une seule fois
# par processus Python. C'est suffisant pour un pipeline batch.
_model: SentenceTransformer | None = None
_index = None


def _get_resources():
    """Lazy loading des ressources coûteuses (modèle 400MB, connexion Pinecone)."""
    global _model, _index
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    if _index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        _index = pc.Index(INDEX_NAME)
    return _model, _index


def _build_retrieval_query(competence: str, type_seance: str, niveau_cible: str | None) -> str:
    """
    Construit une requête optimisée pour le retrieval de chunks pédagogiques.

    Principe : on veut des chunks qui parlent de la TECHNIQUE et des CRITÈRES
    d'enseignement, pas des conditions réglementaires d'accès aux brevets.
    On enrichit donc la requête avec des termes pédagogiques.
    """
    type_mots = {
        "initiation": "découverte introduction premiers pas technique",
        "remediation": "erreurs fréquentes difficultés correction conseils",
        "perfectionnement": "maîtrise perfectionnement critères évaluation progression",
    }
    mots_pedagogiques = type_mots.get(type_seance, "technique éducatif")
    niveau_str = f"niveau {niveau_cible}" if niveau_cible else ""
    return f"{competence} {mots_pedagogiques} {niveau_str} plongée FFESSM".strip()


def retrieve_chunks(state: EducatifState) -> dict:
    """
    Nœud LangGraph — récupère les extraits MFT pertinents pour la compétence.

    Stratégie de filtrage :
    - Si niveau_cible connu → filtre Pinecone sur niveau + "Général" (règles transversales)
    - Sinon → pas de filtre, on prend les 6 meilleurs toutes sources confondues
    """
    try:
        model, index = _get_resources()

        competence = state.get("competence") or ""
        niveau_cible = state.get("niveau_cible")
        type_seance = state.get("type_seance", "initiation")

        query = _build_retrieval_query(competence, type_seance, niveau_cible)
        vector = model.encode(query).tolist()

        query_kwargs = {
            "vector": vector,
            "top_k": N_RESULTS,
            "include_metadata": True,
        }

        # Filtre niveau — même logique que app.py : on inclut "Général"
        # car les règles fédérales transversales s'appliquent à tous les niveaux
        if niveau_cible:
            query_kwargs["filter"] = {"niveau": {"$in": [niveau_cible, "Général"]}}

        results = index.query(**query_kwargs)
        chunks = [m["metadata"]["text"] for m in results["matches"]]

        return {"retrieved_chunks": chunks}

    except Exception as e:
        return {"retrieved_chunks": [], "error": f"RAGRetriever: {e}"}
