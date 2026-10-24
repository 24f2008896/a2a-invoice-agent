
from ai_client import client, MODEL

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": 'Reply with exactly this JSON: {"status":"ok"}',
        }
    ],
    response_format={"type": "json_object"},
    temperature=0,
)

print("Model:", MODEL)
print("Response:", response.choices[0].message.content)
