"""
Interface web du chatbot FFESSM MFT — Streamlit

Streamlit fonctionne comme ça :
- Le script est ré-exécuté de haut en bas à chaque interaction
- st.session_state persiste les données entre les exécutions (l'historique du chat ici)
- Les widgets (st.chat_input, st.chat_message) gèrent l'UI
"""
import os
import streamlit as st
import anthropic
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "ffessm-mft"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
N_RESULTS = 12
HISTORY_WINDOW = 3  # nb de tours de conversation passés injectés dans le prompt

NIVEAU_KEYWORDS = {
    "niveau 1": "N1", "n1": "N1", "pe20": "N1",
    "niveau 2": "N2", "n2": "N2", "pa20": "N2", "pe40": "N2",
    "niveau 3": "N3", "n3": "N3", "pa40": "N3", "pe60": "N3", "pa60": "N3",
    "niveau 4": "N4", "n4": "N4", "guide de palanquée": "N4",
    "niveau 5": "N5", "n5": "N5",
    "initiateur": "Initiateur",
    "mf1": "MF1", "moniteur 1": "MF1",
    "mf2": "MF2", "moniteur 2": "MF2",
    "nitrox": "Nitrox", "trimix": "Trimix",
    "rifap": "RIFAP", "sidemount": "Sidemount",
}


def detect_niveau(question: str) -> str | None:
    q = question.lower()
    for keyword, tag in NIVEAU_KEYWORDS.items():
        if keyword in q:
            return tag
    return None


# ─── Chargement des ressources (une seule fois grâce au cache Streamlit) ──────

@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index(INDEX_NAME)


def get_claude():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ─── Query rewriting ──────────────────────────────────────────────────────────

def rewrite_query(question: str, history: list[dict]) -> str:
    """
    Reformule la question en tenant compte de l'historique de conversation.

    Problème sans ça :
      Tour 1 : "conditions d'accès au recycleur avec déco ?"
      Tour 2 : "et les prérequis ?" → embedder "et les prérequis ?" tout seul
               → la recherche ne sait pas qu'on parle du recycleur

    Solution : on demande à Claude (appel rapide, non streamé) de produire
    une question autonome qui incorpore le contexte.
      → "Quels sont les prérequis pour la formation recycleur circuit fermé
         avec décompression FFESSM ?"
    """
    if not history:
        return question

    # On ne garde que les N derniers tours pour le contexte
    recent = history[-(HISTORY_WINDOW * 2):]
    history_text = "\n".join(
        f"{'Utilisateur' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
        for m in recent
    )

    prompt = f"""Voici une conversation sur les formations FFESSM :

{history_text}

Nouvelle question : {question}

Si la nouvelle question est une suite ou une référence à la conversation (pronoms, "ça", "ce sujet", question courte sans contexte), reformule-la en une question autonome et complète qui incorpore le contexte nécessaire.
Si la question est déjà autonome et claire, retourne-la telle quelle.
Retourne UNIQUEMENT la question reformulée, sans explication."""

    claude = get_claude()
    response = claude.messages.create(
        model="claude-haiku-4-5-20251001",  # modèle rapide pour le rewriting
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ─── Pipeline RAG ─────────────────────────────────────────────────────────────

def build_messages(question: str, contexts: list[str], history: list[dict]) -> list[dict]:
    """
    Construit les messages pour Claude avec :
    - L'historique de conversation (HISTORY_WINDOW derniers tours)
    - Le contexte MFT (chunks Pinecone) injecté dans le premier message système
    - La question actuelle
    """
    context_block = "\n\n---\n\n".join(contexts)
    system_context = f"""Tu es un assistant expert en plongée sous-marine et formations FFESSM.
Réponds en te basant UNIQUEMENT sur les extraits du Manuel de Formation Technique (MFT) fournis.
Si la réponse n'est pas dans les extraits, dis-le clairement sans inventer.

EXTRAITS MFT PERTINENTS :
{context_block}"""

    messages = []

    # Historique des tours précédents
    for msg in history[-(HISTORY_WINDOW * 2):]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Question actuelle
    messages.append({"role": "user", "content": question})

    return system_context, messages


def ask(question: str, history: list[dict]) -> tuple:
    model = load_model()
    index = get_pinecone_index()

    # 1. Rewrite la question avec le contexte conversationnel
    search_query = rewrite_query(question, history)

    # 2. Recherche vectorielle
    question_vector = model.encode(search_query).tolist()
    niveau = detect_niveau(question) or detect_niveau(search_query)
    query_kwargs = {
        "vector": question_vector,
        "top_k": N_RESULTS,
        "include_metadata": True,
    }
    if niveau:
        query_kwargs["filter"] = {"niveau": {"$eq": niveau}}

    results = index.query(**query_kwargs)
    matches = results["matches"]

    if not matches:
        return (lambda: iter(["Aucun extrait pertinent trouvé dans les MFTs."])), [], search_query

    contexts = [m["metadata"]["text"] for m in matches]

    # 3. Génération avec Claude en streaming + historique
    claude = get_claude()
    system_context, messages = build_messages(question, contexts, history)

    def stream_response():
        with claude.messages.stream(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=system_context,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    # 4. Sources
    seen, sources = set(), []
    for m in matches:
        meta = m["metadata"]
        key = (meta["source"], meta.get("section", ""), meta.get("page", ""))
        if key not in seen:
            seen.add(key)
            label = meta["source"]
            if meta.get("section"):
                label += f" › {meta['section']}"
            if meta.get("page"):
                label += f" (p.{int(meta['page'])})"
            sources.append(label)

    return stream_response, sources, search_query


# ─── UI ───────────────────────────────────────────────────────────────────────

SUGGESTED_QUESTIONS = [
    "Conditions d'accès au Niveau 2 ?",
    "Prérogatives du Guide de Palanquée (N4) ?",
    "Quelles sont les compétences requises pour le MF1 ?",
    "Différence entre PA40 et PE40 ?",
    "Conditions pour plonger en autonomie à 60 m ?",
]

st.set_page_config(
    page_title="FFESSM MFT — Assistant formation",
    page_icon="🤿",
    layout="centered",
)

with st.sidebar:
    st.markdown("## 🤿 Assistant MFT FFESSM")
    st.markdown("Posez vos questions sur les formations et brevets de plongée FFESSM.")
    st.divider()
    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Basé sur les Manuels de Formation Technique (MFT) FFESSM.")

st.title("🤿 Assistant MFT FFESSM")
st.caption("Posez vos questions sur les formations et brevets de plongée FFESSM")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Suggestions si aucun message
if not st.session_state.messages:
    st.markdown("**Questions fréquentes :**")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(q, key=f"suggestion_{i}", use_container_width=True):
            st.session_state._suggested = q
            st.rerun()

# Question suggérée sélectionnée
if hasattr(st.session_state, "_suggested") and st.session_state._suggested:
    question_from_suggestion = st.session_state._suggested
    st.session_state._suggested = None
else:
    question_from_suggestion = None

# Affiche l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# Input utilisateur
_chat_input = st.chat_input("Ex : Quelles sont les conditions pour le Niveau 2 ?")
question = question_from_suggestion or _chat_input
if question:

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # On passe l'historique sans le dernier message (la question courante)
        history = st.session_state.messages[:-1]

        stream_fn, sources, rewritten = ask(question, history)

        # Affiche la query reformulée si elle diffère de l'originale
        niveau = detect_niveau(question) or detect_niveau(rewritten)
        if niveau:
            st.caption(f"🔍 Recherche filtrée sur : **{niveau}**")
        if rewritten.strip().lower() != question.strip().lower():
            st.caption(f"🔄 Question reformulée : *{rewritten}*")

        full_response = st.write_stream(stream_fn)

        if sources:
            with st.expander("📚 Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })
