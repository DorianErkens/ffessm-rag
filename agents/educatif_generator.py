"""
Node 4 — Educatif Generator

Rôle : générer la fiche éducatif complète et structurée à partir de :
  - la question originale du moniteur
  - les métadonnées extraites (compétence, niveau, type de séance)
  - les extraits MFT récupérés par le RAGRetriever

C'est le nœud le plus "intelligent" du pipeline : il doit synthétiser des
informations hétérogènes et produire un document pédagogique cohérent.
→ Modèle : Claude Sonnet (pas Haiku, on a besoin de raisonnement réel ici)

Technique : Tool Use (Structured Output via function calling)
─────────────────────────────────────────────────────────────
Au lieu de demander à Claude de "retourner un JSON" (fragile : guillemets non
échappés, trailing commas, texte parasite), on utilise le mécanisme tool_use
de l'API Anthropic :

  1. On définit un "outil fictif" generate_fiche_educatif avec un JSON Schema
     qui décrit exactement la structure attendue.
  2. On force Claude à appeler cet outil avec tool_choice={"type": "any"}.
  3. L'API garantit que la réponse est un JSON valide respectant le schéma.

Avantages vs prompt JSON :
  ✅ JSON toujours valide (validé côté API Anthropic)
  ✅ Pas de guillemets non échappés, pas de trailing commas
  ✅ Typage fort (listes vs strings, champs obligatoires vs optionnels)
  ✅ Plus robuste sur des réponses longues (éducatifs détaillés)
"""
import os

try:
    from agents.state import EducatifState
except ImportError:
    from state import EducatifState  # type: ignore

from anthropic import Anthropic


# ─── Schéma de l'outil (JSON Schema) ─────────────────────────────────────────
# C'est la "signature de fonction" que Claude doit remplir.
# Chaque propriété est typée et décrite — Claude utilise ces descriptions
# pour comprendre ce qu'il faut mettre dans chaque champ.

FICHE_TOOL = {
    "name": "generate_fiche_educatif",
    "description": "Génère une fiche éducatif structurée pour un moniteur FFESSM",
    "input_schema": {
        "type": "object",
        "properties": {
            "contexte": {
                "type": "string",
                "description": "Description de la situation d'enseignement en 2-3 phrases"
            },
            "objectif": {
                "type": "string",
                "description": "Ce que l'élève doit être capable de faire à l'issue de la séance"
            },
            "position_formation": {
                "type": "string",
                "description": "Où cette compétence s'inscrit dans le cursus FFESSM (avant quoi, après quoi)"
            },
            "justification": {
                "type": "string",
                "description": "Pourquoi cette compétence est fondamentale à maîtriser à ce niveau"
            },
            "prerequis": {
                "type": "object",
                "properties": {
                    "techniques": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Liste des prérequis techniques"
                    },
                    "securite": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Points de sécurité importants à vérifier avant la séance"
                    }
                },
                "required": ["techniques", "securite"]
            },
            "educatifs": {
                "type": "array",
                "description": "Liste des exercices progressifs (3 à 5 éducatifs)",
                "items": {
                    "type": "object",
                    "properties": {
                        "ordre": {"type": "integer", "description": "Numéro d'ordre (1, 2, 3...)"},
                        "titre": {"type": "string", "description": "Nom court de l'exercice"},
                        "description": {"type": "string", "description": "Description détaillée et actionnable"},
                        "milieu": {"type": "string", "description": "Où pratiquer : bassin / petit fond / pleine eau"},
                        "duree_estimee": {"type": "string", "description": "Durée estimée ex: 10 min"},
                        "critere_reussite": {"type": "string", "description": "Comment sait-on que c'est réussi ?"}
                    },
                    "required": ["ordre", "titre", "description", "milieu", "critere_reussite"]
                }
            },
            "evaluation": {
                "type": "string",
                "description": "Comment le moniteur évalue l'acquisition de la compétence en fin de séance"
            },
            "pour_aller_plus_loin": {
                "type": "string",
                "description": "Suggestions d'approfondissement ou de complexification pour les séances suivantes"
            }
        },
        "required": [
            "contexte", "objectif", "position_formation", "justification",
            "prerequis", "educatifs", "evaluation", "pour_aller_plus_loin"
        ]
    }
}

GENERATION_PROMPT = """Tu es un expert en pédagogie de la plongée sous-marine FFESSM.
Tu aides des moniteurs à préparer leurs séances d'enseignement.

CONTEXTE DE LA DEMANDE :
- Compétence ciblée : {competence}
- Niveau des élèves : {niveau_cible}
- Stade actuel de l'élève : {niveau_eleve}
- Type de séance : {type_seance}
  • initiation       = première découverte de la compétence
  • remediation      = corriger un problème identifié
  • perfectionnement = affiner une compétence déjà acquise

QUESTION DU MONITEUR : {question}

EXTRAITS DU MANUEL DE FORMATION TECHNIQUE (MFT) FFESSM :
{chunks}

Génère une fiche éducatif complète pour ce moniteur en appelant l'outil generate_fiche_educatif.

RÈGLES :
- Appuie-toi sur les extraits MFT fournis. S'ils sont insuffisants, utilise tes connaissances FFESSM.
- Adapte la progressivité au type de séance "{type_seance}".
- Propose 3 à 5 éducatifs concrets et applicables en bassin ou en mer."""


def generate_educatif(state: EducatifState) -> dict:
    """
    Nœud LangGraph — génère la fiche éducatif via tool_use.

    tool_choice={"type": "any"} force Claude à utiliser un outil.
    La réponse contient un bloc content de type "tool_use" dont
    l'attribut .input est le dict Python directement désérialisé — pas besoin
    de json.loads(), l'API s'en charge.
    """
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    chunks_text = "\n\n---\n\n".join(state.get("retrieved_chunks") or [])
    if not chunks_text:
        chunks_text = "Aucun extrait MFT trouvé — génère à partir de tes connaissances FFESSM."

    prompt = GENERATION_PROMPT.format(
        competence=state.get("competence") or "non spécifiée",
        niveau_cible=state.get("niveau_cible") or "non précisé",
        niveau_eleve=state.get("niveau_eleve") or "non précisé",
        type_seance=state.get("type_seance") or "initiation",
        question=state["question"],
        chunks=chunks_text,
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2500,
            tools=[FICHE_TOOL],
            tool_choice={"type": "any"},  # force l'utilisation d'un outil
            messages=[{"role": "user", "content": prompt}]
        )

        # Trouver le bloc tool_use dans la réponse
        tool_block = next(
            (b for b in response.content if b.type == "tool_use"),
            None
        )
        if not tool_block:
            raise ValueError("Pas de tool_use dans la réponse")

        # .input est déjà un dict Python valide — aucun parsing nécessaire
        fiche = tool_block.input
        fiche["competence"] = state.get("competence")
        fiche["niveau_cible"] = state.get("niveau_cible")
        fiche["type_seance"] = state.get("type_seance")

        return {"fiche": fiche}

    except Exception as e:
        return {"fiche": None, "error": f"EducatifGenerator: {e}"}
