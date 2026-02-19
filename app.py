import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="ML RAG Assistant", layout="wide")

# ---------------- CUSTOM ANIMATED BACKGROUND ---------------- #
st.markdown("""
<style>

/* Remove default white background */
[data-testid="stAppViewContainer"] {
    background: none;
}

/* Full screen animated background */
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

/* Center floating glass container */
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

/* Bold neon bordered title */
h1 {
    text-align: center;
    font-size: 40px;
    font-weight: 900;
    color: #00ffff;
    padding: 12px;
    border: 2px solid #00ffff;
    border-radius: 14px;
    text-shadow: 0 0 25px #00ffff;
    margin-bottom: 20px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #ffffffcc;
    margin-bottom: 25px;
}
/* White Chat Bubbles */
.stChatMessage {
    background: white !important;
    color: #1e1e1e !important;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 12px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    transition: 0.3s ease;
}

/* Subtle hover glow */
.stChatMessage:hover {
    transform: scale(1.01);
    box-shadow: 0 0 18px rgba(0,255,255,0.3);
}

/* Make assistant bubble slightly elevated */
[data-testid="chat-message-assistant"] {
    background: white !important;
}

/* User bubble slight tint */
[data-testid="chat-message-user"] {
    background: #f5f5f5 !important;
}


/* Hide footer */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)


# ---------------- IMPORT AFTER UI ---------------- #
from rag_pipeline import answer_question

# ---------------- TITLE ---------------- #
st.title("🤖 ML RAG Assistant")
st.markdown("<div class='subtitle'>Reliable Grounded Answers with Intelligent Trust Scoring</div>", unsafe_allow_html=True)

# ---------------- SESSION ---------------- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- CHAT INPUT ---------------- #
user_input = st.chat_input("💬 Ask a Machine Learning question...")

if user_input:
    result = answer_question(user_input)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", result))

# ---------------- CHAT DISPLAY ---------------- #
for role, message in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message["answer"])

            trust = message["final_trust_score"]

            if trust >= 0.75:
                st.success(f"🟢 Trust Score: {trust:.3f} | HIGH TRUST")
            elif trust >= 0.5:
                st.warning(f"🟡 Trust Score: {trust:.3f} | MEDIUM TRUST")
            else:
                st.error(f"🔴 Trust Score: {trust:.3f} | LOW TRUST")
