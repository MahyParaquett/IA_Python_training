import streamlit as st
from langchain.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import MessagesState, StateGraph
from dotenv import load_dotenv
import os 

# ==============================
# CONFIGURAÇÃO INICIAL
# ==============================
st.title("Chatbot com Memória – Aula 2")
st.set_page_config(layout='wide', page_title="Chatbot com Memória", page_icon="🧠")

load_dotenv() 

# ==============================
# MEMÓRIA (SESSION)
# ==============================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Olá! Sou um chatbot com memória. Como posso ajudar você hoje?")]

# ==============================
# MODELO
# ==============================
llm = ChatOpenAI(
    api_key= os.environ.get("OPENROUTER_API_KEY"),
    base_url='https://openrouter.ai/api/v1',
    model="openai/gpt-4o-mini",
    streaming=True
)

# ==============================
# PROMPT
# ==============================
prompt = """
Voce é um assistente virtual de IA qua ajuda os usuários a responder perguntas e fornecer informações. Você tem acesso a uma memória de conversas anteriores, o que permite que você mantenha o contexto e forneça respostas mais precisas.

#Tom de voz
- Seja amigável e prestativo.
- Use uma linguagem clara e simples.
- Se possivel use emjis para torna a conversa mais leve e divertida.
"""

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        ("placeholder", "{chat_history}")
    ]
)

llm_with_prompt = chat_template | llm

# ==============================
# LANGGRAPH
# ==============================
def call_chat(state: MessagesState):
    response = llm_with_prompt.invoke({
         "messages": state["messages"]
    })
    return{
        'messages':[response]
    }

#Montar o graph
graph = StateGraph(MessagesState)
graph.add_node('chat', call_chat)
graph.set_entry_point('chat')
app = graph.compile()

# ==============================
# INPUT DO USUÁRIO
# ==============================

user_input = st.text_input("Digite sua pergunta:")
if user_input:
    # adiciona pergunta no histórico
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    with st.spinner("Pensando... 🤔"):
        result = llm_with_prompt.invoke({
            "chat_history": st.session_state.chat_history
        })
        
    st.session_state.chat_history.append(AIMessage(content=result.content))
    
    
# ==============================
# EXIBIÇÃO DO CHAT
# ==============================
for msg in st.session_state.chat_history:
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)
    else:
        with st.chat_message("user"):
            st.markdown(msg.content)