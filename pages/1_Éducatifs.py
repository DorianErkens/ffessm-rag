"""
Page Streamlit — Module Éducatifs

Point d'entrée UI du pipeline LangGraph.

Architecture de la page :
  1. Validation du persona "moniteur" (bouton explicite)
  2. Si activé : interface de saisie de question pédagogique
  3. Pipeline LangGraph avec affichage pas-à-pas via stream()
  4. Rendu de la fiche éducatif structurée

Pourquoi une page séparée et pas un onglet dans app.py ?
Streamlit multi-pages (dossier pages/) est le pattern natif pour séparer
des fonctionnalités distinctes. Les fichiers dans pages/ sont auto-découverts
par Streamlit — le nom du fichier devient le nom de la page dans la sidebar.
"""
import os
import sys
import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Permet d'importer les modules agents/ depuis la racine du projet
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from agents.graph import educatif_graph
from agents.state import EducatifState


# ─── Config page ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FFESSM — Éducatifs",
    page_icon="🎓",
    layout="centered",
)


# ─── Fonctions de rendu ────────────────────────────────────────────────────────
# Définies en haut pour éviter les erreurs de référence avant définition.

def render_fiche(fiche: dict):
    """Affiche la fiche éducatif structurée."""
    st.divider()

    # En-tête
    type_icons = {
        "initiation": "🟢 Initiation",
        "remediation": "🟠 Remédiation",
        "perfectionnement": "🔵 Perfectionnement",
    }
    type_label = type_icons.get(fiche.get("type_seance", ""), fiche.get("type_seance", ""))
    niveau = fiche.get("niveau_cible", "")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## {fiche.get('competence', '').title()}")
    with col2:
        st.markdown(f"**{type_label}**")
        if niveau:
            st.markdown(f"`{niveau}`")

    # Contexte & Objectif
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Contexte")
        st.markdown(fiche.get("contexte", ""))
    with col_b:
        st.markdown("### Objectif de séance")
        st.markdown(fiche.get("objectif", ""))

    # Position & Justification
    with st.expander("📍 Position dans la formation & justification"):
        st.markdown(f"**Position dans le cursus**\n\n{fiche.get('position_formation', '')}")
        st.markdown(f"**Pourquoi c'est important**\n\n{fiche.get('justification', '')}")

    # Prérequis
    prereqs = fiche.get("prerequis", {})
    if prereqs:
        with st.expander("⚠️ Prérequis"):
            col_t, col_s = st.columns(2)
            with col_t:
                st.markdown("**Techniques**")
                for item in prereqs.get("techniques", []):
                    st.markdown(f"- {item}")
            with col_s:
                st.markdown("**Sécurité**")
                for item in prereqs.get("securite", []):
                    st.markdown(f"- {item}")

    # Éducatifs
    st.markdown("### Éducatifs")
    for edu in fiche.get("educatifs", []):
        label = f"**{edu.get('ordre', '?')}. {edu.get('titre', '')}**  ·  _{edu.get('milieu', '')}_"
        with st.expander(label):
            st.markdown(edu.get("description", ""))
            col_d, col_c = st.columns(2)
            with col_d:
                if edu.get("duree_estimee"):
                    st.caption(f"⏱ {edu['duree_estimee']}")
            with col_c:
                st.success(f"✅ {edu.get('critere_reussite', '')}")

    # Évaluation
    st.markdown("### Évaluation")
    st.markdown(fiche.get("evaluation", ""))

    # Pour aller plus loin
    with st.expander("🚀 Pour aller plus loin"):
        st.markdown(fiche.get("pour_aller_plus_loin", ""))

    # Export brut
    with st.expander("📋 JSON brut (debug)"):
        st.code(json.dumps(fiche, ensure_ascii=False, indent=2), language="json")


def run_pipeline_with_progress(question: str) -> EducatifState:
    """
    Lance le pipeline LangGraph en mode stream() et affiche la progression.

    stream() vs invoke() :
    - invoke() → bloquant, retourne le state final en une fois (simple)
    - stream() → retourne les states intermédiaires nœud par nœud, en temps réel

    On utilise stream() ici pour montrer à l'utilisateur que le pipeline
    avance : intention détectée → compétence extraite → MFT recherché → fiche générée.
    C'est la vraie valeur ajoutée de LangGraph : observabilité native du flux.
    """
    initial_state: EducatifState = {
        "question": question,
        "intent": None, "competence": None, "niveau_cible": None,
        "niveau_eleve": None, "type_seance": None,
        "retrieved_chunks": [], "fiche": None, "error": None,
    }

    status = st.empty()
    final_state = dict(initial_state)

    for step in educatif_graph.stream(initial_state):
        node_name = list(step.keys())[0]
        node_output = step[node_name]
        final_state = {**final_state, **node_output}

        if node_name == "intent":
            intent = node_output.get("intent")
            if intent == "enseigner":
                status.info("✅ Question pédagogique détectée — extraction des paramètres...")
            else:
                status.warning(
                    "ℹ️ Question factuelle détectée. "
                    "Consultez l'[Assistant MFT](/) pour une réponse réglementaire."
                )

        elif node_name == "competency":
            comp = node_output.get("competence", "?")
            niv = node_output.get("niveau_cible") or "non précisé"
            typ = node_output.get("type_seance", "?")
            status.info(
                f"✅ **{comp}** · Niveau **{niv}** · Type **{typ}** — "
                f"Recherche dans le MFT..."
            )

        elif node_name == "retriever":
            n = len(node_output.get("retrieved_chunks", []))
            status.info(f"✅ {n} extraits MFT trouvés — génération de la fiche...")

        elif node_name == "generator":
            status.success("✅ Fiche générée !")

    return final_state


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎓 Générateur d'Éducatifs")
    st.markdown("Module pédagogique pour les **moniteurs FFESSM**.")
    st.divider()
    st.caption(
        "Pipeline : Classificateur d'intention → Extracteur de compétence "
        "→ RAG MFT → Générateur de fiche"
    )
    st.divider()

    if st.button("🔄 Nouvelle fiche", use_container_width=True):
        for key in ["moniteur_mode", "fiche_result", "last_question", "_pending_question"]:
            st.session_state.pop(key, None)
        st.rerun()

    # Debug panel
    if st.session_state.get("fiche_result"):
        result = st.session_state["fiche_result"]
        with st.expander("🔬 Debug — State LangGraph"):
            st.json({
                "intent": result.get("intent"),
                "competence": result.get("competence"),
                "niveau_cible": result.get("niveau_cible"),
                "type_seance": result.get("type_seance"),
                "chunks_count": len(result.get("retrieved_chunks", [])),
                "error": result.get("error"),
            })


# ─── Init session state ────────────────────────────────────────────────────────

st.session_state.setdefault("moniteur_mode", False)
st.session_state.setdefault("fiche_result", None)
st.session_state.setdefault("last_question", None)


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🎓 Générateur d'Éducatifs FFESSM")
st.caption("Préparez vos séances d'enseignement avec l'aide de l'IA")
st.divider()


# ─── Validation persona moniteur ───────────────────────────────────────────────

if not st.session_state.moniteur_mode:
    st.info(
        "Ce module est destiné aux **moniteurs et cadres FFESSM**. "
        "Il génère des fiches éducatifs pédagogiques basées sur le MFT."
    )
    st.markdown("###")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            "🎓 Je suis moniteur — activer le module",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.moniteur_mode = True
            st.rerun()
    st.markdown("###")
    st.caption("Pour les questions réglementaires, utilisez l'[Assistant MFT](/).")
    st.stop()


# ─── Interface moniteur ────────────────────────────────────────────────────────

st.success("✅ Mode moniteur activé")
st.markdown(
    "Posez votre question pédagogique : compétence à enseigner, "
    "problème d'un élève, préparation de séance..."
)

EXEMPLES = [
    "Mon élève N1 a du mal à palmer sans remonter",
    "Éducatifs pour introduire l'équilibrage en N2",
    "Comment corriger un élève qui remonte trop vite ?",
    "Progressions pour amener un PE20 à maîtriser la RSE",
]

st.markdown("**Exemples :**")
cols = st.columns(2)
for i, ex in enumerate(EXEMPLES):
    if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
        st.session_state["_pending_question"] = ex
        st.rerun()

question_input = st.chat_input("Décrivez votre besoin pédagogique...")
question = st.session_state.pop("_pending_question", None) or question_input

if question:
    st.session_state.last_question = question
    st.session_state.fiche_result = None

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        result = run_pipeline_with_progress(question)
        st.session_state.fiche_result = result

    if result.get("fiche"):
        render_fiche(result["fiche"])
    elif result.get("error"):
        st.error(f"Erreur pipeline : {result['error']}")

# Ré-affiche la fiche si déjà générée (navigation Streamlit)
elif st.session_state.fiche_result and st.session_state.fiche_result.get("fiche"):
    with st.chat_message("user"):
        st.markdown(st.session_state.last_question or "")
    render_fiche(st.session_state.fiche_result["fiche"])
