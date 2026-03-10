from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import os 
import uuid
import re

load_dotenv() 

model = ChatOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini"
)

workflow = StateGraph(state_schema=MessagesState)

def extract_name(text):
    match = re.search(r"(me chamo|meu nome é)\s+([A-Za-zÀ-ÿ]+)", text, re.IGNORECASE)
    if match:
        return match.group(2)
    return None

def call_model(state: MessagesState):
    messages = state["messages"]

    last_user_text = messages[-1].content
    name = extract_name(last_user_text)

    if name:
        # Instrução para o modelo lembrar e usar o nome
        messages = [
            SystemMessage(content=f"O nome da usuária é {name}. Lembre-se disso e use o nome nas próximas respostas."),
            *messages
        ]

    response = model.invoke(messages)
    return {"messages": response}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

thread_id = uuid.uuid4()
config = {"configurable": {"thread_id": thread_id}}

# Loop de conversa
while True:
    user_input = input("Você: ")
    if user_input.lower() in ["sair", "exit", "quit"]:
        break

    input_messages = [HumanMessage(user_input)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
