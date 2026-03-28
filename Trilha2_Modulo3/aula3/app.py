import streamlit as st
from langchain.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.message import MessagesState, StateGraph
from dotenv import load_dotenv
import os 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# ==============================
# CONFIGURAÇÃO INICIAL ChromaDB
# ==============================
embeddings = OpenAIEmbeddings(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url='https://openrouter.ai/api/v1',
    model="openai/text-embedding-3-small"
)

vectorDB = Chroma(
    collection_name="meus_documentos",
    embedding_function=embeddings,
    persist_directory="./chroma_store"
)

retriever = vectorDB.as_retriever(search_kwargs={"k": 4})

# ==============================
# CONFIGURAÇÃO INICIAL Página
# ==============================
st.title("Chatbot com Memória + RAG – Aula 3")
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
Você é um assistente com memória e acesso a documentos.

Use as informações abaixo para responder:

<contexto>
{context}
</contexto>

# Regras:
- Use o contexto acima se for relevante
- Se não tiver resposta no contexto, responda normalmente
- Seja claro, amigável e útil
- Use emojis se fizer sentido 😊
"""

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
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
     st.session_state.chat_history.append(HumanMessage(content=user_input))

     with st.spinner("Buscando informação... 🔍"):
        docs = retriever.invoke(user_input)
        contexto = "\n\n".join([doc.page_content for doc in docs])

     with st.spinner("Pensando... 🤔"):
        response = llm_with_prompt.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input,
            "context": contexto
        })

     st.session_state.chat_history.append(AIMessage(content=response.content))
     
    
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