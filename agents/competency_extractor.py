"""
Node 2 — Competency Extractor

Rôle : à partir de la question du moniteur, extraire 4 informations structurées :
  - competence   : la compétence ciblée ("palmation", "équilibrage", "RSE"...)
  - niveau_cible : le niveau de certification de l'élève (N1, N2, N3, MF1...)
  - niveau_eleve : le stade de l'élève dans son apprentissage (optionnel)
  - type_seance  : initiation | remediation | perfectionnement

Pourquoi ce nœud séparé de l'IntentClassifier ?
Single Responsibility : le classifieur répond à "est-ce pédagogique ?"
Cet extracteur répond à "sur quoi et pour qui ?". Deux questions différentes,
deux nœuds différents → plus facile à tester, à modifier, à déboguer indépendamment.

Modèle : Haiku (extraction d'entités = tâche structurée, pas de raisonnement complexe)

Technique : structured extraction via JSON + règles de déduction du type de séance
  - initiation   → première introduction à la compétence
  - remediation  → corriger un problème identifié
  - perfectionnement → affiner une compétence déjà acquise
"""
import json
import os

try:
    from agents.state import EducatifState
except ImportError:
    from state import EducatifState  # type: ignore

from anthropic import Anthropic


EXTRACTION_PROMPT = """Tu es un expert en formations FFESSM. Analyse cette question d'un moniteur.

Extrais les informations suivantes :

1. "competence" : la compétence technique ciblée (en minuscules, ex: "palmation", "équilibrage",
   "gestion de l'air", "remontée sécurisée", "lâcher de parachute", "vidage de masque")
   Si non explicite, déduis-la du contexte.

2. "niveau_cible" : le niveau FFESSM des élèves. Valeurs possibles :
   N1, N2, N3, N4, MF1, MF2, Initiateur, Nitrox, Sidemount
   Si PE20 → N1, PA20 → N2, PE40 → N2, PA40 → N3, PE60 → N3
   Si non mentionné → null

3. "niveau_eleve" : le degré de maîtrise actuel de l'élève (optionnel)
   Valeurs : "découverte", "en progression", "presque acquis"
   Déduis-le des indices dans la question ("du mal à", "commence à", "presque")
   Si non mentionné → null

4. "type_seance" : le type d'intervention pédagogique
   - "initiation"        → premier contact avec la compétence ("introduire", "première fois",
                           "découvrir", "aborder", "débuter")
   - "remediation"       → corriger un problème identifié ("du mal à", "problème avec",
                           "n'y arrive pas", "erreur", "difficultés", "corriger")
   - "perfectionnement"  → affiner ce qui est déjà acquis ("améliorer", "affiner",
                           "progresser", "maîtriser mieux", "amener à", "progressivement")
   Si ambiguë → "initiation" (valeur par défaut la plus sûre)

EXEMPLES :
Q: "Mon élève N1 a du mal à palmer sans remonter" →
   {{"competence": "palmation", "niveau_cible": "N1", "niveau_eleve": "en progression", "type_seance": "remediation"}}

Q: "Comment introduire l'équilibrage à des plongeurs N2 qui débutent ?" →
   {{"competence": "équilibrage", "niveau_cible": "N2", "niveau_eleve": "découverte", "type_seance": "initiation"}}

Q: "Comment amener progressivement un PE20 à maîtriser la RSE ?" →
   {{"competence": "remontée sécurisée", "niveau_cible": "N1", "niveau_eleve": null, "type_seance": "perfectionnement"}}

Q: "Éducatifs pour travailler la gestion de l'air chez des N2 ?" →
   {{"competence": "gestion de l'air", "niveau_cible": "N2", "niveau_eleve": null, "type_seance": "initiation"}}

Réponds UNIQUEMENT par un JSON valide, sans texte autour.

Question : {question}"""


def extract_competency(state: EducatifState) -> dict:
    """
    Nœud LangGraph — extrait les métadonnées pédagogiques de la question.

    Note sur la robustesse : si l'extraction échoue, on retourne des valeurs
    par défaut plutôt que de bloquer le pipeline. Le Node 4 (générateur)
    saura gérer une compétence "inconnue" mieux qu'une exception.
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": EXTRACTION_PROMPT.format(question=state["question"])
            }]
        )

        text = response.content[0].text.strip()
        start, end = text.find("{"), text.rfind("}") + 1
        if start == -1:
            raise ValueError(f"Pas de JSON dans : {text}")

        data = json.loads(text[start:end])

        # Normalisation du niveau (PE20 → N1, etc.) — double sécurité si le
        # modèle retourne le format brut malgré les instructions
        NIVEAU_MAP = {
            "pe20": "N1", "pa20": "N2", "pe40": "N2",
            "pa40": "N3", "pe60": "N3", "pa60": "N3",
        }
        niveau = data.get("niveau_cible")
        if niveau:
            niveau = NIVEAU_MAP.get(niveau.lower(), niveau)

        type_seance = data.get("type_seance", "initiation")
        if type_seance not in ("initiation", "remediation", "perfectionnement"):
            type_seance = "initiation"

        return {
            "competence": data.get("competence") or "compétence non identifiée",
            "niveau_cible": niveau,
            "niveau_eleve": data.get("niveau_eleve"),
            "type_seance": type_seance,
        }

    except Exception as e:
        return {
            "competence": "compétence non identifiée",
            "niveau_cible": None,
            "niveau_eleve": None,
            "type_seance": "initiation",
            "error": f"CompetencyExtractor: {e}",
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dotenv import load_dotenv
    load_dotenv()

    CAS_TESTS = [
        ("Mon élève N1 a du mal à palmer sans remonter",
         {"competence": "palmation", "niveau_cible": "N1", "type_seance": "remediation"}),
        ("Comment introduire l'équilibrage à des N2 qui débutent ?",
         {"competence": "équilibrage", "niveau_cible": "N2", "type_seance": "initiation"}),
        ("Comment amener progressivement un PE20 à maîtriser la RSE ?",
         {"competence": "remontée sécurisée", "niveau_cible": "N1", "type_seance": "perfectionnement"}),
        ("Éducatifs pour travailler la gestion de l'air chez des N2 ?",
         {"competence": "gestion de l'air", "niveau_cible": "N2", "type_seance": "initiation"}),
    ]

    print("=== Test de l'extracteur de compétences ===\n")
    for question, attendu in CAS_TESTS:
        state: EducatifState = {
            "question": question, "intent": "enseigner",
            "competence": None, "niveau_cible": None, "niveau_eleve": None,
            "type_seance": None, "retrieved_chunks": [], "fiche": None, "error": None,
        }
        result = extract_competency(state)
        print(f"Q: {question}")
        for key, val_attendue in attendu.items():
            val_obtenue = result.get(key)
            ok = "✅" if val_obtenue == val_attendue else "❌"
            print(f"  {ok} {key}: attendu={val_attendue!r} | obtenu={val_obtenue!r}")
        print()
