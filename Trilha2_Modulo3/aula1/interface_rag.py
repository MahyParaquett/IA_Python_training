import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 

load_dotenv() 
model = ChatOpenAI(
    api_key= os.environ.get("OPENROUTER_API_KEY"),
    base_url='https://openrouter.ai/api/v1',
    model="openai/gpt-4o-mini",
    streaming=True
)

st.title("Meu sistema RAG")
st.subheader("Sistema de IA com LangChain + OpenAI")
pergunta = st.text_area("Digite sua pergunta:")

if st.button("Enviar"):
    if not pergunta:
        st.warning("Digite uma pergunta primeiro!")
    else:
        with st.spinner("Pensando... 🤔"):
            resposta = model.invoke(pergunta)

        st.success("Resposta gerada!")
        st.write("### Resposta:")
        st.write(resposta.content)