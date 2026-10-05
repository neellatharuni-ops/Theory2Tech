import os
import serpapi
from dotenv import load_dotenv

load_dotenv()


def search_google(query: str):
    """Search Google through SerpApi."""

    api_key = os.getenv("SERPAPI_KEY")

    if not api_key:
        raise ValueError(
            "SERPAPI_KEY is missing. Add it to the .env file."
        )

    client = serpapi.Client(
        api_key=api_key
    )

    results = client.search({
        "engine": "google",
        "q": query,
        "hl": "en",
        "gl": "in",
        "num": 10,
        "safe": "active",
    })

    return results.get(
        "organic_results",
        []
    )