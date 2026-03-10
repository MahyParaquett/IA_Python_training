from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
from chatbot import app

# Configuração visual

st.set_page_config(layout='wide', page_title='Chatbot de loja de bicicletas', page_icon='🚴')
st.title("Loja de Bicicletas - Assistente Virtual")

# Histórico de mensagens inicial

if "message_history" not in st.session_state:
 st.session_state.message_history = [
    AIMessage(content="Olá! 🚴 Sou seu assistente virtual da loja de bicicletas. Como posso te ajudar?")
 ]

# Exibição das mensagens na interface
for msg in st.session_state.message_history:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)

# Campo de entrada
user_input = st.chat_input("Digite aqui...")

# Se houver mensagem do usuário
if user_input:
    # 1) adiciona msg do usuário
    st.session_state.message_history.append(HumanMessage(content=user_input))

    # 2) mostra a msg do usuário imediatamente
    with st.chat_message("user"):
        st.markdown(user_input)

    # 3) chama o app e pega resposta
    response = app.invoke({"messages": st.session_state.message_history})

    # 4) atualiza histórico com retorno do LangChain
    st.session_state.message_history = response["messages"]

    # 5) mostra a última resposta (a nova do assistant)
    last_msg = st.session_state.message_history[-1]
    with st.chat_message("assistant"):
        st.markdown(last_msg.content)