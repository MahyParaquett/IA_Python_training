from langchain_openai import ChatOpenAI
from langgraph.graph import  StateGraph
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os 

# loading variables from .env file
load_dotenv() 

# Prompt com tom de voz e orientações
template = """
Você é um assistente de IA que responde a pergunta abaixo com base no contexto fornecido. Se a pergunta não puder ser respondida com o contexto, diga que não sabe. Seja amigável e útil.
{context}

{question}

Answer:
"""

# Template de prompt estruturado
chat_com_contexto_template = ChatPromptTemplate.from_template(template)

#Definir o modelo de LLM a ser utilizado 
model = ChatOpenAI(
    api_key= os.environ.get("OPENROUTER_API_KEY"),
    base_url='https://openrouter.ai/api/v1',
    model="openai/gpt-4o-mini",
    streaming=True
)

# Pipeline entre prompt e modelo
llm_with_prompt = chat_com_contexto_template| model

# Estado da aplicação
class State(dict):
   question: str
   context: list
   answer: str
   
# Função de resposta
def generate(state: State):
    docs_content = "\n\n".join(doc["page_content"] for doc in state["context"])

    chunks = llm_with_prompt.stream(
        {"question": state["question"], "context": docs_content}
    )

    full_text = ""
    for chunk in chunks:
        # chunk é AIMessageChunk (normalmente tem .content)
        full_text += chunk.content or ""

    return {"answer": full_text}

# Criando grafo do LangGraph
graph = StateGraph(State)
graph.add_node("generate", generate)
graph.set_entry_point("generate")
graph.set_finish_point("generate")

# Compilando a aplicação
app = graph.compile()
