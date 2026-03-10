import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from chatbot import app

st.set_page_config(layout="wide", page_title="Chatbot de leitor de documentos", page_icon="🚴")
st.title("Leitor de Documentos - Assistente Virtual")

# Histórico inicial
if "message_history" not in st.session_state:
    st.session_state.message_history = [
        AIMessage(content="Olá! 🚴 Sou seu assistente virtual de leitura de documentos. Envie um PDF ou pergunte algo!")
    ]

# PDF
uploaded_file = st.file_uploader("Faça o upload de um PDF para análise", type=["pdf"])
pdf_text = ""

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    pdf_text = "\n\n".join(doc.page_content for doc in docs)

# 1) Sempre renderiza o histórico
for msg in st.session_state.message_history:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# Input
user_input = st.chat_input("Digite aqui...")

if user_input:
    # 2) Salva e mostra mensagem do usuário
    st.session_state.message_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # 3) Placeholder do assistente (stream)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        partial = ""

        # IMPORTANTE: use messages para token-a-token (se seu chatbot.py estiver no formato runnable)
        for event in app.stream(
            {"question": user_input, "context": [{"page_content": pdf_text}]},
            stream_mode="messages",
        ):
            msg = event[0] if isinstance(event, tuple) else event
            chunk = getattr(msg, "content", "")

            if chunk:
                partial += chunk
                placeholder.markdown(partial)

        # 4) Salva a resposta final no histórico (isso impede “sumir”)
        st.session_state.message_history.append(AIMessage(content=partial))