import requests
import os
from dotenv import load_dotenv
os.makedirs("dados", exist_ok=True)
os.makedirs("chroma_store", exist_ok=True)
from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


#-------------------------
# ETAPA 1: Baixar os documentos da internet e salvar localmente
#-------------------------

pdf_url = "https://arxiv.org/pdf/1706.03762.pdf" #Attention is All You Need 
html_url = "http://www.gutenberg.org/files/11/11-h/11-h.htm" #Alice no País das Maravilhas

pdf_path = "dados/attention_is_all_you_need.pdf"
html_path = "dados/alice_no_pais_das_maravilhas.html"

#Baixar PDF
r = requests.get(pdf_url, timeout=60)
r.raise_for_status() #verificar se a requisição foi bem sucedida
with open(pdf_path, "wb") as f:
    f.write(r.content)

#Baixar HTML
r = requests.get(html_url, timeout=60)
r.raise_for_status() #verificar se a requisição foi bem sucedida
with open(html_path, "wb") as f:
    f.write(r.content)

pdf_path, html_path


#-------------------------
# ETAPA 2: Carregar o conteúdo dos arquivos usando os loaders do LangChain
#-------------------------
#Carregando o conteúdo dos arquivos usando os loaders do LangChain, 
# ele permite extrair o texto de forma estruturada, mantendo a formatação e os metadados, 
# como títulos e parágrafos.

pdf_loader = PyPDFLoader(file_path=pdf_path)
docs_pdf = pdf_loader.load()
docs_pdf[0]

html_loader = BSHTMLLoader(html_path, open_encoding="utf-8",bs_kwargs={"features": "html.parser"})
docs_html = html_loader.load()
len(docs_html), docs_html[0].metadata.get("title")

#-------------------------
# ETAPA 3: Quebrar o conteudo em partes menores, os Churks
#-------------------------

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
docs = splitter.split_documents(docs_pdf + docs_html)
len(docs), docs[0].metadata

#-------------------------
# ETAPA 4: Transformar textos em vetores usando os embbedings
#-------------------------
load_dotenv() 

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

ids = vectorDB.add_documents(docs)

#-------------------------
# ETAPA 5: Teste da massa de dados criada, fazendo uma consulta e verificando os resultados
#-------------------------

consulta = "Who is Alice?"
resultados = vectorDB.similarity_search(consulta, k=3)
for d in resultados:
    print(d.metadata, d.page_content[:200], "\n---")