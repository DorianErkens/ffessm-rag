"""
State LangGraph du pipeline éducatifs.

Le State est le "contrat de données" qui circule entre tous les nœuds du graphe.
Chaque nœud reçoit le state complet, fait son travail, et retourne un dict
avec SEULEMENT les clés qu'il a modifiées — LangGraph fusionne automatiquement.

Pourquoi TypedDict et pas dataclass ou Pydantic ?
LangGraph est conçu pour TypedDict : il peut les sérialiser/désérialiser
pour le checkpointing (reprise après erreur) et l'inspection des traces.
"""
from typing import TypedDict, Literal, Optional


class EducatifState(TypedDict):
    # ── Input utilisateur ──────────────────────────────────────────────────────
    question: str                           # question brute du moniteur

    # ── Node 1 : IntentClassifier ─────────────────────────────────────────────
    intent: Optional[Literal["info", "enseigner"]]
    # "info"      → question réglementaire/factuelle → on reste sur le RAG classique
    # "enseigner" → intention pédagogique → on continue vers les éducatifs

    # ── Node 2 : CompetencyExtractor ──────────────────────────────────────────
    competence: Optional[str]               # ex: "palmation", "équilibrage"
    niveau_cible: Optional[str]             # ex: "N1", "N2", "MF1"
    niveau_eleve: Optional[str]             # ex: "débutant", "confirmé" (optionnel)
    type_seance: Optional[Literal["initiation", "remediation", "perfectionnement"]]

    # ── Node 3 : RAGRetriever ─────────────────────────────────────────────────
    retrieved_chunks: list                  # extraits MFT pertinents (liste de str)

    # ── Node 4 : EducatifGenerator ────────────────────────────────────────────
    fiche: Optional[dict]                   # fiche éducatif complète structurée

    # ── Méta ──────────────────────────────────────────────────────────────────
    error: Optional[str]                    # propagation d'erreurs entre nœuds
