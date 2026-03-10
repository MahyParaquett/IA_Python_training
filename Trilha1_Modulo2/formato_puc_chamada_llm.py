from openai import OpenAI
client = OpenAI(
  base_url="http://139.82.24.30:1234/v1",
  api_key="lm-studio"
)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Você é um expert em história dos LLMs."},
        {"role": "user", "content": "Conte uma história sobre o desenvolvimento da Inteligência Artificial até a invenção dos LLMs."}
    ],
    temperature=0.8
)
print(completion.choices[0].message.content)