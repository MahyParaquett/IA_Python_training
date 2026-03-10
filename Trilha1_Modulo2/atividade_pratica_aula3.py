from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import os 
import uuid

# loading variables from .env file
load_dotenv() 

#Definir o modelo de LLM a ser utilizado 
model = ChatOpenAI(
    api_key= os.environ.get("OPENROUTER_API_KEY"),
    base_url='https://openrouter.ai/api/v1',
    model="openai/gpt-4o-mini"
)

#Definir o grafo de estados
workflow = StateGraph(state_schema=MessagesState)

#Como chamaremos o modelo com as mendagens
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

#Definindo os estados da conversa
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

#Armazenar a memoria da conversa
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

#Definindo o ID da conversa pela thread
thread_id = uuid.uuid4()
config={"configurable": {"thread_id": thread_id}}

# Primeira mensagem
query = "olá, eu sou a Mahyara."
input_messages = [HumanMessage(query)]

output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()

# Segunda mensagem (testando memória)
query = "Como eu me chamo?"
input_messages = [HumanMessage(query)]

output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()