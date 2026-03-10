from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import getpass
import os


if not os.environ.get('OPENAI_API_KEY'):
    os.environ['OPENAI_API_KEY'] = getpass.getpass('Enter your OpenAI API key: ')
    
llm = ChatOpenAI(model="gpt-4o-mini")

resposta = llm.invoke([
    SystemMessage(content="Traduza o seguinte texto de Inglês para Português."),
    HumanMessage(content="Hi, I'm OpenIa.")
])

print(resposta.content)

# O código está correto porém o uso da API da OpenIa é pago, 
# e como não tenho crédito da erro 401 no console