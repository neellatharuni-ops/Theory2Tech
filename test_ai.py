import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "OPENAI_API_KEY is missing from .env"
    )


client = OpenAI(api_key=api_key)


response = client.responses.create(
    model="gpt-5-mini",
    input="Explain B-Tree in simple terms for a CSE student."
)


print("\nAI RESPONSE:\n")
print(response.output_text)