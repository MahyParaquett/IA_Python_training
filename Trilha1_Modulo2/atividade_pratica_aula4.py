from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os 
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import  StateGraph
from langchain_core.prompts import PromptTemplate


# loading variables from .env file
load_dotenv() 

#Definir o modelo de LLM a ser utilizado 
model = ChatOpenAI(
    api_key= os.environ.get("OPENROUTER_API_KEY"),
    base_url='https://openrouter.ai/api/v1',
    model="openai/gpt-4o-mini"
)

#Carregar o documento em PDF
loader = PyPDFLoader("Estado-do-clima-no-Brasil-em-2022-OFICIAL.pdf")
docs = loader.load()

#Define o estado do grafo, que é um dicionário com as chaves "pergunta", "contexto" e "resposta" define o formato de dados que vai circular no grafo.
class State(dict):
    pergunta: str
    contexto: list
    resposta: str

def generate(state:State):
    #Concatenar todo o texto 
    docs_content = "\n\n".join([doc.page_content for doc in state["contexto"]])
    
    #Passa o template para o modelo, preenchendo com a pergunta e o contexto concatenado
    formatted_prompt = prompt.format(
        pergunta=state["pergunta"],
        contexto=docs_content
    )
    
    response = model.invoke(formatted_prompt)
    
    return {"resposta": response.content}

#Esse é o prompt que será enviado ao modelo
template = """
Responda a pergunta abaixo com base no contexto fornecido
{contexto}

{pergunta}

Answer:
"""

#Construção do grafo (Start → generate → End)
prompt = PromptTemplate(
    template=template,
    input_variables=["pergunta", "contexto"]
)

graph_builder = StateGraph(State)
graph_builder.add_node("generate", generate) #Chamada do coração da execução, onde o modelo é chamado para gerar a resposta

graph_builder.set_entry_point("generate")
graph_builder.set_finish_point("generate")

app = graph_builder.compile()

contexto = docs

pergunta = "Qual é o principal argumento do documento?"

##Rodar Grafo/Execução
response = app.invoke({"pergunta": pergunta, "contexto": contexto})
print(response["resposta"])