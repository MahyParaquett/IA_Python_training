from openai import OpenAI

client = OpenAI(
    api_key='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',
    base_url='https://openrouter.ai/api/v1'
)

completion = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {
            'role': 'system', 
            'content': 'Você é um expert sobre a história dos LLMs'
        },
        {
            'role': 'user',
            'content': 'Escreva uma história sobre o desenvolvimento do campo da Inteligência Artificial até a invenção dos LLMs.'
        }
    ],
    temperature=0.8,
    max_tokens=400
)

print(completion.choices[0].message)