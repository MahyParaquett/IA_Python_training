from openai import OpenAI
import os 
from dotenv import load_dotenv

# loading variables from .env file
load_dotenv() 

 
client = OpenAI(
    api_key= os.environ.get("OPENROUTER_API_KEY"),
    base_url='https://openrouter.ai/api/v1'
)

completion = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {
            'role': 'system', 
            'content': 'Traduza o seguinte texto de Inglês para Português.'
        },
        {
            'role': 'user',
            'content': 'Hi, Im OpenIA'
        }
    ],
    temperature=0.3
)

print(completion.choices[0].message.content)