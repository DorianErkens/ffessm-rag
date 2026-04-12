"""
Graph assembly — le cœur du pipeline LangGraph

C'est ici que les 4 nœuds sont connectés en un graphe orienté avec
des arêtes conditionnelles. C'est le fichier le plus pédagogique du module.

─── Comment LangGraph fonctionne ────────────────────────────────────────────

1. StateGraph(EducatifState) : crée un graphe dont chaque nœud partage
   le même état typé. Le state circule de nœud en nœud.

2. add_node("nom", fonction) : enregistre un nœud. La fonction doit avoir
   la signature : (state: dict) -> dict (retourne les clés modifiées)

3. add_edge(A, B) : arête inconditionnelle — B est toujours appelé après A

4. add_conditional_edges(A, routeur, {valeur: nœud}) : arête conditionnelle.
   Le "routeur" est une fonction qui lit le state et retourne une string
   qui détermine le prochain nœud. C'est le mécanisme de branchement.

5. set_entry_point("nom") : quel nœud lance le pipeline

6. compile() : valide le graphe et retourne un Runnable (objet invocable)

─── Le flow ici ─────────────────────────────────────────────────────────────

                    ┌─────────────────┐
                    │  intent_node    │  ← Haiku : "info" ou "enseigner" ?
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ intent == "info"             │ intent == "enseigner"
              ▼                              ▼
            END                    ┌─────────────────┐
     (pas d'éducatifs)             │ competency_node │  ← Haiku : extrait compétence/niveau
                                   └────────┬────────┘
                                            │
                                   ┌────────▼────────┐
                                   │  retriever_node │  ← Pinecone : chunks MFT
                                   └────────┬────────┘
                                            │
                                   ┌────────▼────────┐
                                   │ generator_node  │  ← Sonnet : génère la fiche
                                   └────────┬────────┘
                                            │
                                           END
"""
from langgraph.graph import StateGraph, END

try:
    from agents.state import EducatifState
    from agents.intent_classifier import classify_intent
    from agents.competency_extractor import extract_competency
    from agents.rag_retriever import retrieve_chunks
    from agents.educatif_generator import generate_educatif
except ImportError:
    from state import EducatifState  # type: ignore
    from intent_classifier import classify_intent  # type: ignore
    from competency_extractor import extract_competency  # type: ignore
    from rag_retriever import retrieve_chunks  # type: ignore
    from educatif_generator import generate_educatif  # type: ignore


def _route_intent(state: EducatifState) -> str:
    """
    Routeur conditionnel — lit l'intent dans le state et retourne le nom
    du prochain nœud. LangGraph appelle cette fonction après intent_node
    pour décider où aller.

    Retourne une string qui doit correspondre aux clés du dict passé à
    add_conditional_edges (ou à la constante END).
    """
    if state.get("intent") == "enseigner":
        return "competency"
    return END


def build_graph():
    """
    Construit et compile le graphe LangGraph.

    On appelle cette fonction une seule fois et on réutilise le graphe compilé
    (même pattern que st.cache_resource dans Streamlit).
    """
    graph = StateGraph(EducatifState)

    # ── Enregistrement des nœuds ──────────────────────────────────────────────
    graph.add_node("intent", classify_intent)
    graph.add_node("competency", extract_competency)
    graph.add_node("retriever", retrieve_chunks)
    graph.add_node("generator", generate_educatif)

    # ── Arêtes ────────────────────────────────────────────────────────────────
    graph.set_entry_point("intent")

    # Arête conditionnelle après intent : enseigner → suite, info → stop
    graph.add_conditional_edges(
        "intent",
        _route_intent,
        {
            "competency": "competency",  # si enseigner
            END: END,                    # si info
        }
    )

    # Arêtes inconditionnelles pour la suite du pipeline
    graph.add_edge("competency", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", END)

    return graph.compile()


# ─── Singleton du graphe compilé ──────────────────────────────────────────────
# Compiler le graphe a un coût (validation, construction de l'exécuteur).
# On le fait une seule fois au chargement du module.
educatif_graph = build_graph()


def run_educatif_pipeline(question: str) -> EducatifState:
    """
    Point d'entrée public du pipeline.

    Construit l'état initial et invoque le graphe.
    Retourne le state final avec tous les champs remplis par les nœuds.

    invoke() vs stream() :
    - invoke() → bloquant, retourne le state final complet. Simple.
    - stream() → retourne les states intermédiaires au fur et à mesure.
      Utile pour afficher la progression dans l'UI Streamlit.
      On utilisera stream() dans la page Streamlit pour l'effet "loading step by step".
    """
    initial_state: EducatifState = {
        "question": question,
        "intent": None,
        "competence": None,
        "niveau_cible": None,
        "niveau_eleve": None,
        "type_seance": None,
        "retrieved_chunks": [],
        "fiche": None,
        "error": None,
    }

    return educatif_graph.invoke(initial_state)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    import json

    print("=== Test du pipeline complet ===\n")

    # Test 1 : question pédagogique → doit produire une fiche
    q1 = "Mon élève N1 a du mal à palmer sans remonter, quels éducatifs ?"
    print(f"Q1 (enseigner): {q1}")
    result1 = run_educatif_pipeline(q1)
    print(f"  intent      : {result1['intent']}")
    print(f"  competence  : {result1['competence']}")
    print(f"  niveau      : {result1['niveau_cible']}")
    print(f"  type_seance : {result1['type_seance']}")
    print(f"  chunks RAG  : {len(result1['retrieved_chunks'])} extraits")
    print(f"  fiche       : {'✅ générée' if result1['fiche'] else '❌ absente'}")
    if result1.get("error"):
        print(f"  error       : {result1['error']}")

    print()

    # Test 2 : question info → doit s'arrêter après intent
    q2 = "Conditions d'accès au Niveau 2 ?"
    print(f"Q2 (info): {q2}")
    result2 = run_educatif_pipeline(q2)
    print(f"  intent      : {result2['intent']}")
    print(f"  competence  : {result2['competence']} (doit être None)")
    print(f"  fiche       : {result2['fiche']} (doit être None)")
