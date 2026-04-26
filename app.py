import streamlit as st

# ================== PAGE CONFIG (MUST BE FIRST) ==================
st.set_page_config(page_title="ML RAG Assistant", layout="wide")

# ================== CUSTOM ANIMATED BACKGROUND ==================
st.markdown("""
<style>

/* Remove default background */
[data-testid="stAppViewContainer"] {
    background: none;
}

/* Full animated gradient background */
html, body, .stApp {
    height: 100%;
    margin: 0;
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1a2980);
    background-size: 400% 400%;
    animation: gradientMove 18s ease infinite;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass center container */
.block-container {
    max-width: 750px;
    margin: 60px auto;
    padding: 30px;
    background: rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(25px);
    border-radius: 25px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 0 60px rgba(0,255,255,0.25);
}

/* Neon bordered title */
h1 {
    text-align: center;
    font-size: 38px;
    font-weight: 900;
    color: #00ffff;
    padding: 12px;
    border: 2px solid #00ffff;
    border-radius: 14px;
    text-shadow: 0 0 25px #00ffff;
    margin-bottom: 15px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 15px;
    color: #ffffffcc;
    margin-bottom: 25px;
}

/* White chat bubbles */
.stChatMessage {
    background: white !important;
    color: #1e1e1e !important;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 12px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    transition: 0.3s ease;
}

/* User slightly tinted */
[data-testid="chat-message-user"] {
    background: #f5f5f5 !important;
}

/* Hover glow */
.stChatMessage:hover {
    transform: scale(1.01);
    box-shadow: 0 0 18px rgba(0,255,255,0.3);
}

/* Hide footer */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# ================== IMPORT AFTER UI ==================
from rag_pipeline import answer_question

# ================== ABSTENTION THRESHOLD ==================
TRUST_THRESHOLD = 0.5          # Below this → "I don't know"
COVERAGE_THRESHOLD = 0.1       # If sentence coverage is near zero → abstain


# ================== HELPER: Should the system abstain? ==================
def should_abstain(result: dict) -> bool:
    """
    Returns True if the answer is too unreliable to show.
    Two conditions trigger abstention:
      1. Final trust score is below threshold
      2. Sentence coverage is near zero (nothing grounded in retrieved docs)
    """
    trust = result.get("final_trust_score", 0)
    coverage = result.get("sentence_coverage", 0)

    if coverage <= COVERAGE_THRESHOLD:
        return True
    if trust < TRUST_THRESHOLD:
        return True
    return False


# ================== HELPER: Render one assistant response ==================
def render_assistant_response(result: dict):
    """Renders the answer (or abstention message) + metrics + trust badge."""

    abstained = should_abstain(result)

    if abstained:
        # ── Abstention message ──────────────────────────────────────────────
        st.warning(
            "🤷 **I don't know.** "
            "I couldn't find reliable information in my knowledge base to answer this. "
            "Please try rephrasing or ask a different ML-related question."
        )
        st.caption(
            f"ℹ️ Abstained because: "
            f"Trust Score = {result['final_trust_score']:.3f} "
            f"(threshold {TRUST_THRESHOLD}) | "
            f"Sentence Coverage = {result.get('sentence_coverage', 0):.3f} "
            f"(threshold {COVERAGE_THRESHOLD})"
        )
    else:
        # ── Normal answer ───────────────────────────────────────────────────
        st.write(result["answer"])

    # ── Metrics (always shown so user can inspect) ──────────────────────────
    st.markdown("**Metrics:**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Semantic Similarity",  f"{result.get('semantic_similarity',  'N/A')}")
    m2.metric("Sentence Coverage",    f"{result.get('sentence_coverage',    'N/A')}")
    m3.metric("Concept Overlap",      f"{result.get('concept_overlap',      'N/A')}")
    m4.metric("Retrieval Confidence", f"{result.get('retrieval_confidence', 'N/A')}")

    # ── Trust badge ──────────────────────────────────────────────────────────
    trust = result["final_trust_score"]
    if trust >= 0.75:
        st.success(f"🟢 Trust Score: {trust:.3f} | HIGH TRUST")
    elif trust >= 0.5:
        st.warning(f"🟡 Trust Score: {trust:.3f} | MEDIUM TRUST")
    else:
        st.error(f"🔴 Trust Score: {trust:.3f} | LOW TRUST — Abstained")


# ================== TITLE ==================
st.title("🤖 ML RAG Assistant")
st.markdown(
    "<div class='subtitle'>Reliable Grounded Answers with Intelligent Trust Scoring</div>",
    unsafe_allow_html=True
)

# ================== SESSION STATE ==================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ================== DISPLAY OLD CHAT ==================
for role, message in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.write(message)
    else:
        with st.chat_message("assistant"):
            render_assistant_response(message)


# ================== CHAT INPUT ==================
user_input = st.chat_input("💬 Ask a Machine Learning question...")

if user_input:

    # 1️⃣ Save + show user message
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)

    # 2️⃣ Get answer from RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            result = answer_question(user_input)

        # 3️⃣ Render response (with abstention logic inside)
        render_assistant_response(result)

    # 4️⃣ Save bot response
    st.session_state.chat_history.append(("bot", result))

    st.rerun()