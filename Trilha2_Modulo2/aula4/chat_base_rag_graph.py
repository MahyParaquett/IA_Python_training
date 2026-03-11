from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from RAG_system import retriever
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from langchain_core.documents import Document
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

# Estado do grafo
# O TypedDict é uma forma de definir a estrutura de um dicionário, especificando as chaves e os tipos dos valores associados a essas chaves. Ele é útil para garantir que o dicionário tenha uma estrutura consistente e para fornecer informações de tipo para ferramentas de análise estática de código.
class GraphState(TypedDict):
    pergunta: str
    docs: List[Document] #define listas tipadas
    contexto: str
    resposta: str

# Nó 1: recuperar documentos
def retrieve_node(state: GraphState) -> GraphState:
    pergunta = state["pergunta"]
    docs = retriever.invoke(pergunta)
    contexto = format_docs(docs)

    return {
        **state,
        "docs": docs,
        "contexto": contexto
    }

# Nó 2: gerar resposta
def generate_node(state: GraphState) -> GraphState:
    pergunta = state["pergunta"]
    contexto = state["contexto"]

    mensagens = template.invoke({
        "pergunta": pergunta,
        "contexto": contexto
    })

    out = model.invoke(mensagens)

    return {
        **state,
        "resposta": out.content
    }

# Montagem do grafo
graph_builder = StateGraph(GraphState)

graph_builder.add_node("retrieve", retrieve_node)
graph_builder.add_node("generate", generate_node)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

# Execução
resultado = graph.invoke({
    "pergunta": "Where is Wonderland?",
    "docs": [],
    "contexto": "",
    "resposta": ""
})

print(resultado["resposta"])