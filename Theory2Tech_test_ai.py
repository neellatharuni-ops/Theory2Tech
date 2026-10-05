import os
from dotenv import load_dotenv

load_dotenv()

print("\n=== Theory2Tech AI API Test v2 ===\n")


# =========================================================
# 1. GOOGLE GEMINI
# =========================================================
print("1. Testing Google Gemini...")

try:
    from google import genai

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    client = genai.Client(api_key=key)

    # Gemini 3.6 Flash is the current stable model for this test.
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: Gemini connection successful."
    )

    text = getattr(response, "text", None)

    if not text:
        raise RuntimeError("Gemini returned no text.")

    print("✅ Gemini:", text.strip())

except Exception as e:
    print("❌ Gemini failed:", type(e).__name__, "-", e)


# =========================================================
# 2. GROQ
# =========================================================
print("\n2. Testing Groq...")

try:
    from groq import Groq

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing from .env")

    client = Groq(api_key=key)

    # Don't hard-code an old/deprecated model.
    # Ask Groq which models are currently available to this API key.
    models = client.models.list()

    available = [
        model.id
        for model in models.data
        if getattr(model, "active", True)
    ]

    if not available:
        raise RuntimeError("Groq returned no active models.")

    preferred = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    ]

    model = next(
        (m for m in preferred if m in available),
        available[0],
    )

    print("   Using Groq model:", model)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Groq connection successful."
            }
        ],
        max_tokens=30,
    )

    text = response.choices[0].message.content

    if not text:
        raise RuntimeError("Groq returned no text.")

    print("✅ Groq:", text.strip())

except Exception as e:
    print("❌ Groq failed:", type(e).__name__, "-", e)


# =========================================================
# 3. HUGGING FACE INFERENCE PROVIDERS
# =========================================================
print("\n3. Testing Hugging Face...")

try:
    from huggingface_hub import InferenceClient

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing from .env")

    # Hugging Face automatically chooses an available provider.
    # This is the official OpenAI-compatible chat-completions route.
    model = os.getenv(
        "HF_MODEL",
        "openai/gpt-oss-120b:fastest",
    )

    client = InferenceClient(
        api_key=token,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Hugging Face connection successful."
            }
        ],
        max_tokens=30,
    )

    if not response.choices:
        raise RuntimeError("Hugging Face returned no choices.")

    text = response.choices[0].message.content

    if not text:
        raise RuntimeError(
            "Hugging Face returned an empty text response. "
            "The selected provider/model may not support this request."
        )

    print("   Using Hugging Face model:", model)
    print("✅ Hugging Face:", text.strip())

except Exception as e:
    print("❌ Hugging Face failed:", type(e).__name__, "-", e)


print("\n=== Test complete ===")
print("Your API keys are never printed by this script.")
