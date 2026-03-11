#import base
from dotenv import load_dotenv
import os 
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv() 

def carregar_retriever():
      #Embeddings + Vetores
      #Chroma.from_documents(...) → cria/popula o banco // Chroma(...) → abre um banco já existente
    
    embeddings = OpenAIEmbeddings(
        api_key= os.environ.get("OPENROUTER_API_KEY"),
        base_url='https://openrouter.ai/api/v1',
        model="openai/text-embedding-3-small"
    )

    
    vectorDB = Chroma(
        collection_name="meus_documentos",
        embedding_function=embeddings,
        persist_directory="./chroma_store"
    )
    
    #Retriever
    retriever = vectorDB.as_retriever(search_kwargs={"k":3}) #recuperar os 3 chunks mais relevantes
    return retriever

retriever = carregar_retriever()
