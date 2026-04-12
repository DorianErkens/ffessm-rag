"""
Node 1 — Intent Classifier

Rôle : décider si la question est pédagogique ("enseigner") ou factuelle ("info").
C'est la bifurcation principale du graphe — si elle est fausse, tout le reste l'est.

Modèle choisi : Claude Haiku
Pourquoi : classification binaire = tâche simple. Haiku est 10x moins cher que
Sonnet et 3x plus rapide. Garder Sonnet pour les tâches de raisonnement (Node 4).

Technique utilisée : few-shot prompting + JSON forcé
Le few-shot (exemples dans le prompt) réduit drastiquement les erreurs de
classification sur des cas ambigus. Le JSON forcé évite le parsing fragile
de réponses en langage naturel.
"""
import json
import os
from anthropic import Anthropic
try:
    # Import absolu — fonctionne quand importé depuis graph.py (racine du projet)
    from agents.state import EducatifState
except ImportError:
    # Import relatif — fonctionne quand lancé directement : python agents/intent_classifier.py
    from state import EducatifState  # type: ignore


# ─── Prompt de classification ─────────────────────────────────────────────────
# Le few-shot est volontairement équilibré : 4 exemples "enseigner", 4 "info",
# avec des cas ambigus inclus pour que le modèle apprenne les nuances.

CLASSIFICATION_PROMPT = """Tu es un expert en pédagogie de la plongée sous-marine FFESSM.

Analyse la question d'un moniteur et détermine son intention :

- "enseigner" : le moniteur veut enseigner, corriger ou faire progresser un élève
  → signaux : "comment enseigner", "éducatifs pour", "mon élève a du mal à",
    "exercices pour travailler", "progressions pour", "remédiation", "séance sur"

- "info" : question factuelle ou réglementaire sur les formations FFESSM
  → signaux : "conditions pour", "prérogatives de", "qu'est-ce que", "définition",
    "profondeur max", "quelle règle", "est-ce que c'est obligatoire"

EXEMPLES :
Q: "Comment faire travailler la palmation à un élève N1 ?" → enseigner
Q: "Quels éducatifs pour corriger une mauvaise position en pleine eau ?" → enseigner
Q: "J'ai un élève qui panique lors des remontées, comment remédier ?" → enseigner
Q: "Quelles progressions pédagogiques pour l'équilibrage en N2 ?" → enseigner
Q: "Comment amener progressivement un PE20 à maîtriser la RSE ?" → enseigner
Q: "Conditions d'accès au Niveau 2 FFESSM ?" → info
Q: "Prérogatives du Guide de Palanquée ?" → info
Q: "Quelle est la profondeur maximale pour un PE40 ?" → info
Q: "Différence entre PA40 et PE40 ?" → info

Réponds UNIQUEMENT par un JSON valide, sans texte autour :
{{"intent": "enseigner" ou "info", "confidence": 0.0 à 1.0, "signal": "mot ou phrase clé qui a décidé"}}

Question : {question}"""


def classify_intent(state: EducatifState) -> dict:
    """
    Nœud LangGraph — reçoit le state, retourne les clés modifiées.

    Convention LangGraph : on ne retourne PAS le state complet,
    seulement les clés qu'on a modifiées. Le graphe fusionne automatiquement.
    C'est différent des fonctions classiques — à bien retenir.
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": CLASSIFICATION_PROMPT.format(question=state["question"])
            }]
        )

        text = response.content[0].text.strip()

        # Extraction robuste du JSON — Haiku peut parfois ajouter du texte autour
        start, end = text.find("{"), text.rfind("}") + 1
        if start == -1:
            raise ValueError(f"Pas de JSON dans la réponse : {text}")

        result = json.loads(text[start:end])
        intent = result.get("intent", "info")

        # Validation de la valeur retournée
        if intent not in ("enseigner", "info"):
            intent = "info"

        return {"intent": intent}

    except Exception as e:
        # En cas d'erreur, on defaulte à "info" (comportement sûr : pas d'éducatifs)
        # et on propage l'erreur pour debug
        return {"intent": "info", "error": f"IntentClassifier: {e}"}


# ─── Tests unitaires ──────────────────────────────────────────────────────────
# On les met dans le même fichier car ils testent directement le prompt.
# Lance avec : pytest agents/intent_classifier.py -v
# Ou depuis la racine : pytest agents/ -v

if __name__ == "__main__":
    """
    Mode test rapide sans pytest — utile pour itérer vite sur le prompt.
    Lance avec : python agents/intent_classifier.py (depuis la racine du projet)

    Trick Python : quand on lance un fichier directement (pas comme module),
    son dossier parent n'est pas dans sys.path. On l'ajoute manuellement
    pour que 'from agents.state import' fonctionne.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))  # ajoute la racine

    from dotenv import load_dotenv
    load_dotenv()

    CAS_TESTS = [
        # (question, intent_attendu, description)
        # ── Questions "enseigner" ────────────────────────────────────────────
        ("Comment enseigner la palmation à un élève niveau 1 ?",
         "enseigner", "signal explicite 'enseigner'"),

        ("Quels éducatifs pour travailler l'équilibrage en N2 ?",
         "enseigner", "signal explicite 'éducatifs'"),

        ("Mon élève a du mal à contrôler sa remontée, qu'est-ce que je peux faire ?",
         "enseigner", "signal 'mon élève' + remédiation implicite"),

        ("Je prépare une séance sur la gestion de l'air pour des N1, des idées ?",
         "enseigner", "signal 'séance' + population cible"),

        ("Comment amener progressivement un PE20 à maîtriser la RSE ?",
         "enseigner", "signal 'progressivement' + 'maîtriser'"),

        # ── Questions "info" ─────────────────────────────────────────────────
        ("Quelles sont les conditions d'accès au Niveau 2 FFESSM ?",
         "info", "question réglementaire classique"),

        ("Prérogatives du Guide de Palanquée ?",
         "info", "question sur les droits/niveaux"),

        ("Quelle est la profondeur maximale pour un plongeur PE40 ?",
         "info", "question factuelle sur les limites"),

        ("Différence entre PA40 et PE40 ?",
         "info", "question de définition"),

        ("Est-ce qu'un MF1 peut enseigner jusqu'à 40m ?",
         "info", "question sur les prérogatives moniteur — factuelle"),
    ]

    print("=== Test du classificateur d'intention ===\n")
    total, ok = len(CAS_TESTS), 0
    for question, attendu, desc in CAS_TESTS:
        state: EducatifState = {
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
        result = classify_intent(state)
        intent_obtenu = result.get("intent")
        passed = intent_obtenu == attendu
        ok += passed
        status = "✅" if passed else "❌"
        print(f"{status} [{desc}]")
        print(f"   Q: {question}")
        print(f"   Attendu: {attendu} | Obtenu: {intent_obtenu}")
        print()

    print(f"Résultat : {ok}/{total} ({100*ok//total}%)")
