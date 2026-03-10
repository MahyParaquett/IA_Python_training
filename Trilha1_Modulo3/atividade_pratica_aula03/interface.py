import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from chatbot import app

# Configuração visual
st.set_page_config(layout='wide', page_title='Chatbot de loja de bicicletas', page_icon='🚴')
st.title("Loja de Bicicletas - Assistente Virtual")

# Histórico de mensagens inicial
if "message_history" not in st.session_state:
 st.session_state.message_history = [
    AIMessage(content="Olá! 🚴 Sou seu assistente virtual da loja de bicicletas. Envie um PDF ou pergunte algo!")
 ]

# Upload de arquivo PDF
uploaded_file = st.file_uploader("Faça o upload de um PDF para análise", type=["pdf"])
pdf_text = ""

if uploaded_file is not None:
   with open("temp.pdf", "wb") as f:
    f.write(uploaded_file.read())
   loader = PyPDFLoader("temp.pdf")
   docs = loader.load()
   pdf_text = "\n\n".join(doc.page_content for doc in docs)
   

# Campo de entrada do usuário
user_input = st.chat_input("Digite aqui...")

if user_input:  st.session_state.message_history.append(HumanMessage(content=user_input))
response = app.invoke({
       'question': user_input,
       'context': [{'page_content': pdf_text}]
   })
st.session_state.message_history.append(AIMessage(content=response['answer']))

# Exibição das mensagens na interface
for msg in st.session_state.message_history:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)
