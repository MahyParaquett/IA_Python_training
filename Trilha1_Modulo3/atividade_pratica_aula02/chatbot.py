from langchain_openai import ChatOpenAI
from langgraph.graph import  StateGraph, MessagesState
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import streamlit as st
import os 

# loading variables from .env file
load_dotenv() 

# Prompt com tom de voz e orientações

SYSTEM_PROMPT = ("Você é um assistente de IA que ajuda os usuários a encontrar informações sobre produtos de uma loja de bicicleta. Seja amigável, prestativo e use emojis e trocadilhos. Se não souber, diga que não sabe.")

# Template de chat com sistema + histórico de mensagens

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"), #Injeta o histórico de conversas no prompt
])



#Definir o modelo de LLM a ser utilizado 
model = ChatOpenAI(
    api_key= os.environ.get("OPENROUTER_API_KEY"),
    base_url='https://openrouter.ai/api/v1',
    model="openai/gpt-4o-mini"
)

# Pipeline entre prompt e modelo

llm_with_prompt = chat_prompt | model

# Criando grafo do LangGraph

graph = StateGraph(MessagesState)
def gerar_resposta(state: MessagesState):
 resposta = llm_with_prompt.invoke({"messages": state["messages"]})
 return {"messages": state["messages"] + [resposta]}
graph.add_node("chat", gerar_resposta)
graph.set_entry_point("chat")
graph.set_finish_point("chat")

# Compilando a aplicação

app = graph.compile()
