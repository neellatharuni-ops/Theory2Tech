import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def analyze_concept(concept, search_results):
    """
    Analyze a CSE concept using live SerpApi search results.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. "
            "Add it to the .env file."
        )

    client = OpenAI(api_key=api_key)

    # Prepare the live search information
    sources = []

    for result in search_results[:5]:

        title = result.get("title", "")
        link = result.get("link", "")
        snippet = result.get("snippet", "")

        if title and link:

            sources.append(
                f"Title: {title}\n"
                f"URL: {link}\n"
                f"Description: {snippet}"
            )

    source_text = "\n\n".join(sources)

    prompt = f"""
You are the AI analysis engine for Theory2Tech,
an educational platform that connects Computer Science
concepts with real-world technology.

Analyze the following CSE concept:

CONCEPT:
{concept}

LIVE WEB SOURCES:
{source_text}

Create a clear educational report with these sections:

1. What is it?
Explain the concept simply for a CSE student.

2. Why is it taught?
Explain why this concept matters in Computer Science education.

3. Real-world applications
Explain where this concept is actually used.

4. Industry connection
Mention technologies, systems, databases, companies,
or engineering areas where appropriate.

5. What can I build?
Suggest one practical beginner/intermediate project
using this concept.

6. Related concepts
List important concepts the student should learn next.

7. Key takeaway
Give a short summary of why the concept matters.

Important rules:
- Use the supplied web sources as evidence.
- Do not invent specific facts that are not supported.
- Clearly distinguish general computer-science knowledge
  from information suggested by the sources.
- Keep the explanation understandable for a student.
- Do not include unnecessary introductory text.
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return response.output_text
