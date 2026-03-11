from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from RAG_system import retriever
from dotenv import load_dotenv
import os 

load_dotenv() 
model = ChatOpenAI(
    api_key= os.environ.get("OPENROUTER_API_KEY"),
    base_url='https://openrouter.ai/api/v1',
    model="openai/gpt-4o-mini",
    streaming=True
)

# Prompt com tom de voz e orientações
template = ChatPromptTemplate.from_template(
    "Você é um assistente factual. Use EXCLUSIVAMENTE o contexto para responder.\n"
    "Se não houver informação suficiente, diga isso explicitamente.\n\n"
    "Pergunta:{pergunta}\n\n"
    "Contexto:\n{contexto}\n\n"
    "Responda de forma concisa e cite as fontes no final no formato {{fonte: titulo}}."
)

def format_docs(docs):
    blocos=[]
    for doc in docs:
        titulo = doc.metadata.get("title")
        blocos.append(f"[{titulo}] {doc.page_content}")
    return "\n\n---\n\n".join(blocos)

def answer_linear(pergunta: str):
    docs = retriever.invoke(pergunta)
    contexto = format_docs(docs)
    mensagem = template.format(pergunta=pergunta, contexto=contexto)
    out = model.invoke(mensagem)
    return out.content

print(answer_linear("Where is Wonderland?"))