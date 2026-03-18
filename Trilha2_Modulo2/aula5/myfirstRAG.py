#import base
from dotenv import load_dotenv
import os 
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv() 

def criar_retriever():
    # Coleta de páginas
    topics = ["Lei Geral de Proteção de Dados", "LangChain", "Wikipedia"] 

    documents = []
    for topic in topics:
        loader = WikipediaLoader(query=topic, load_max_docs=1) #1 artigo por tópico
        documents.extend(loader.load())

    #Dividir em chunks, pedaços de texto menores
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120, add_start_index=True) #tamanho do chunk e sobreposição
    chunks = splitter.split_documents(documents)

    #Embeddings + Vetores
    embeddings = OpenAIEmbeddings(
        api_key= os.environ.get("OPENROUTER_API_KEY"),
        base_url='https://openrouter.ai/api/v1',
        model="openai/text-embedding-3-small"
    )

    
    vectorDB = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="wikipedia_pt_rag",
        persist_directory="./chroma_wiki"
    )
    
    #Nas versões mais recentes do Chroma, não é mais necessário chamar o método persist() para salvar os vetores no disco
   # vectorDB.persist() #salvar os vetores no disco

    #Retriever
    #PARAMETRO K define a quantidade de documentos mais relevantes
    #retriever = vectorDB.as_retriever(search_kwargs={"k":4}) #recuperar os 4 chunks mais relevantes
    
    #PARAMETRO SCORE_THRESHOLD define um limite mínimo de relevância para os documentos recuperados
    #retriever = vectorDB.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.7}) #recuperar os 4 chunks mais relevantes
    
    #PARAMETRO DE METADADOS define quais metadados dos documentos serão retornados junto com o conteúdo
    retriever = vectorDB.as_retriever(search_kwargs={"filter": {"fonte": "Wikipedia"}})
    
    return retriever

retriever = criar_retriever()
