import os 
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(question, context):
    prompt = f"""
You are an AI assistant answering questions about a book.

Use ONLY the context below to answer the question.

If the answer is not in the context, say:
"The book does not mention this."

Context:
{context}

Question:
{question}

Answer:
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()
