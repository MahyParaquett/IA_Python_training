from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
import os 

# loading variables from .env file
load_dotenv() 

#Definir o modelo de LLM a ser utilizado 
model = ChatOpenAI(
    api_key= os.environ.get("OPENROUTER_API_KEY"),
    base_url='https://openrouter.ai/api/v1',
    model="openai/gpt-4o-mini"
)

#Configuração da interface
st.title("Chat com IA usando LangChain")
prompt = st.text_area("Digite aqui sua pergunta...")

if st.button("Enviar"):
    if prompt:
        resposta = model.invoke(prompt)
        st.write("**Resposta:**", resposta.content)
    else:
        st.warning("Digiite uma pergunta antes de enviar.")