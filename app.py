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
N_RESULTS = 8

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
    """
    @st.cache_resource : exécuté une seule fois, résultat mis en cache.
    Évite de recharger le modèle d'embedding (300MB) à chaque question.
    """
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def get_pinecone_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index(INDEX_NAME)


def build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    return f"""Tu es un assistant expert en plongée sous-marine et formations FFESSM.
Réponds à la question en te basant UNIQUEMENT sur les extraits du Manuel de Formation Technique (MFT) fournis ci-dessous.
Si la réponse n'est pas dans les extraits, dis-le clairement.

EXTRAITS MFT :
{context_block}

QUESTION : {question}

RÉPONSE :"""


def ask(question: str) -> tuple[str, list[str]]:
    model = load_model()
    index = get_pinecone_index()

    # Recherche vectorielle avec filtrage niveau si détecté
    question_vector = model.encode(question).tolist()
    niveau = detect_niveau(question)
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
        return "Aucun extrait pertinent trouvé dans les MFTs.", []

    contexts = [m["metadata"]["text"] for m in matches]

    # Génération avec Claude en streaming
    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # st.write_stream consomme un générateur → on yield les tokens au fur et à mesure
    def stream_response():
        with claude.messages.stream(
            model="claude-opus-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": build_prompt(question, contexts)}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    # Sources
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

    return stream_response, sources


# ─── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FFESSM MFT — Assistant formation",
    page_icon="🤿",
    layout="centered",
)

st.title("🤿 Assistant MFT FFESSM")
st.caption("Posez vos questions sur les formations et brevets de plongée FFESSM")

# Initialise l'historique du chat dans la session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affiche l'historique
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# Input utilisateur
if question := st.chat_input("Ex : Quelles sont les conditions pour le Niveau 2 ?"):

    # Affiche la question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Génère et affiche la réponse en streaming
    with st.chat_message("assistant"):
        niveau = detect_niveau(question)
        if niveau:
            st.caption(f"🔍 Recherche filtrée sur : **{niveau}**")

        stream_fn, sources = ask(question)

        # write_stream affiche les tokens au fur et à mesure et retourne le texte complet
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
