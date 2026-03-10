from openai import OpenAI #importa a biblioteca OpenAI
client = OpenAI() #Instancia o cliente

#a cada mensansagem é um prompt
response = client.responses.create(
    model="gpt-4.1-mini",
    input="Escreva uma história sobre o desenvolvimento do campo de Inteligência Artificial até a invenção dos LLMs."
)

print(response.output_text)

#Para executar esse código, basta no terminal rodar: python cahamado_llm.py