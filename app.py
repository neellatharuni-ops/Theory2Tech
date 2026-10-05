import os
import re
import json
import http.client
from urllib.parse import urlparse, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import streamlit as st
import serpapi
from dotenv import load_dotenv

# Gemini is the confirmed working AI provider for Theory2Tech.
# It is imported lazily in call_gemini() so the app can still start
# and display the syllabus if the SDK is not installed yet.

# ---------------------------------------------------------
# Theory2Tech
# GPCET R23 CSE -> Real-World Technology
# SerpApi only. No OpenAI API.
# ---------------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Theory2Tech",
    page_icon="🌍",
    layout="wide",
)

DATA_FILE = "syllabus_data.json"

# Stop making additional API calls after a network failure during this run.
# This prevents one connection problem from producing a long Streamlit traceback.
if "serpapi_unavailable" not in st.session_state:
    st.session_state.serpapi_unavailable = False

if "serpapi_error_message" not in st.session_state:
    st.session_state.serpapi_error_message = ""

BAD_DOMAINS = {
    "pinterest.com",
    "scribd.com",
    "studocu.com",
    "coursehero.com",
    "quizlet.com",
    "brainly.com",
    "chegg.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "homework.study.com",
    "slideshare.net",
}

SYLLABUS_NOISE = {
    "syllabus",
    "curriculum",
    "course outcome",
    "course outcomes",
    "question bank",
    "semester",
    "regulation",
    "regulations",
    "unit i",
    "unit ii",
    "unit iii",
    "unit iv",
    "unit v",
    "b.tech",
    "btech",
}

EDUCATION_WORDS = {
    "tutorial", "notes", "guide", "documentation", "explained",
    "learn", "lesson", "example", "examples", "implementation",
    "data structure", "algorithm",
}

COURSE_WORDS = {
    "course", "certificate", "certification", "mooc",
    "enroll", "courseware", "learning path",
}

APPLICATION_WORDS = {
    "database", "index", "storage", "filesystem", "file system",
    "search engine", "production", "server", "software",
    "application", "applications", "used", "uses", "indexing",
    "disk", "record", "query", "transaction", "library",
}

PROJECT_WORDS = {
    "implementation", "project", "repository", "repo",
    "source code", "visualization", "simulator", "application",
}

TRUSTED_DEFINITION_DOMAINS = {
    "wikipedia.org": 5,
    "geeksforgeeks.org": 5,
    "programiz.com": 5,
    "ibm.com": 5,
    "oracle.com": 5,
    "redhat.com": 4,
    "tutorialspoint.com": 3,
}

TRUSTED_LEARNING_DOMAINS = {
    "geeksforgeeks.org": 5,
    "programiz.com": 5,
    "w3schools.com": 4,
    "freecodecamp.org": 4,
    "ibm.com": 4,
    "oracle.com": 4,
    "tutorialspoint.com": 3,
}

COURSE_DOMAINS = {
    "coursera.org": 5,
    "edx.org": 5,
    "udemy.com": 4,
    "nptel.ac.in": 5,
    "swayam.gov.in": 5,
    "ocw.mit.edu": 5,
    "khanacademy.org": 4,
}



# ---------------------------------------------------------
# Gemini AI
# ---------------------------------------------------------

def gemini_available():
    return bool(os.getenv("GEMINI_API_KEY"))


def _gemini_model():
    return os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()


def _gemini_models_to_try():
    """Return configured Gemini model first, followed by safe fallbacks."""
    configured = _gemini_model()

    # Gemini 3.5 Flash-Lite is a good low-latency fallback for a student
    # explanation app. The newer 3.8 Flash is also a current stable model.
    fallback_models = [
        configured,
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
    ]

    output = []
    for model in fallback_models:
        model = model.strip()
        if model and model not in output:
            output.append(model)

    return output


def call_gemini(prompt):
    """Call Gemini with retry + model fallback handling.

    A 503 means Gemini is temporarily overloaded/unavailable. It is not
    treated as an invalid API key. We retry with exponential backoff and,
    if necessary, try another currently supported Flash model.
    """
    import time
    import urllib.error
    import urllib.request

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None, "GEMINI_API_KEY is missing from .env."

    last_error = "Gemini request failed."

    for model in _gemini_models_to_try():
        # Use the documented x-goog-api-key header instead of putting the
        # API key in the URL. This is also more reliable with some proxies.
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": 4000
            }
        }

        # Three attempts per model. Delays: 2s, 4s, then fallback model.
        for attempt in range(3):
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "x-goog-api-key": api_key,
                    "User-Agent": "Theory2Tech/1.0",
                    "Connection": "close",
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    raw = response.read().decode("utf-8")
                    data = json.loads(raw)

                candidates = data.get("candidates", [])

                if not candidates:
                    last_error = f"{model} returned no candidates."
                    break

                parts = (
                    candidates[0]
                    .get("content", {})
                    .get("parts", [])
                )

                answer = "\n".join(
                    part.get("text", "")
                    for part in parts
                    if part.get("text")
                ).strip()

                if answer:
                    return answer, None

                last_error = f"{model} returned an empty response."
                break

            except urllib.error.HTTPError as exc:
                status = exc.code

                try:
                    error_body = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    error_body = ""

                if status in (503, 500, 502, 504):
                    last_error = (
                        f"Gemini model {model} returned HTTP {status}."
                    )
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    # Try the next model after transient server failure.
                    break

                if status == 429:
                    last_error = (
                        f"Gemini model {model} returned HTTP 429 "
                        "(rate limit/quota pressure)."
                    )
                    if attempt < 2:
                        time.sleep(3 * (attempt + 1))
                        continue
                    break

                if status in (401, 403):
                    return (
                        None,
                        "Gemini authentication/permission failed. "
                        "Check GEMINI_API_KEY and the Gemini API configuration."
                    )

                if status == 404:
                    # Do not stop the whole app. Try the next model.
                    last_error = f"Gemini model '{model}' was not found."
                    break

                safe_error = re.sub(
                    r"key=[^&\s]+",
                    "key=REDACTED",
                    error_body,
                )
                return None, f"Gemini HTTP {status}: {safe_error[:500]}"

            except (
                http.client.RemoteDisconnected,
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
            ) as exc:
                # The remote server can close an idle/proxied connection
                # before returning an HTTP response. Retry the same model
                # before moving to the next fallback model.
                last_error = (
                    f"Gemini connection closed unexpectedly ({type(exc).__name__})."
                )
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break

            except urllib.error.URLError as exc:
                last_error = f"Could not connect to Gemini: {exc.reason}"
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break

            except TimeoutError:
                last_error = f"Gemini model {model} timed out."
                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue
                break

            except Exception as exc:
                return None, f"{type(exc).__name__}: {exc}"

    return (
        None,
        last_error
        + " Tried the configured model and Gemini Flash fallbacks. "
        "Please try again shortly if Gemini is temporarily unavailable."
    )


def extract_github_search_terms(gemini_answer, topic):
    """Extract Gemini's project-combination hints to guide GitHub search."""
    terms = []
    for line in re.findall(r"^\s*\*\*Uses:\*\*\s*(.+)$", gemini_answer or "", flags=re.MULTILINE):
        clean = re.sub(r"[\[\](){}]", " ", line)
        clean = re.sub(r"\s+", " ", clean).strip()
        if clean and clean.lower() not in {t.lower() for t in terms}:
            terms.append(clean)

    # Gemini is the project-planning layer. The selected topic is always
    # included so GitHub search stays anchored to the user's exact topic.
    search_terms = [topic]
    for term in terms[:2]:
        if term.lower() != topic.lower():
            search_terms.append(term)

    return search_terms


@st.cache_data(ttl=3600, show_spinner=False)
def github_search_repositories(query, num=10):
    """Find public GitHub repositories without consuming SerpApi quota."""
    params = urlencode({
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": min(int(num), 10),
    })
    url = f"https://api.github.com/search/repositories?{params}"

    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2026-03-10",
            "User-Agent": "Theory2Tech/1.0",
        },
    )

    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))

        results = []
        for repo in payload.get("items", []):
            if repo.get("archived") or repo.get("fork"):
                continue
            if not repo.get("html_url"):
                continue
            results.append(repo)
            if len(results) >= 3:
                break
        return results

    except (HTTPError, URLError, TimeoutError, ValueError):
        return []


def render_github_projects(topic, gemini_answer):
    """Show exactly three real GitHub repositories guided by Gemini's ideas."""
    st.divider()
    st.subheader("🔗 Top 3 Relevant GitHub Projects")
    st.caption(
        "Gemini identifies the relevant project direction and previously learned "
        "concepts. Theory2Tech then finds real public GitHub repositories for that direction."
    )

    terms = extract_github_search_terms(gemini_answer, topic)
    primary_query = " ".join(f'"{term}"' if " " in term else term for term in terms)

    repositories = github_search_repositories(primary_query, 10)

    # If the combined query is too restrictive, fall back to the exact topic.
    if not repositories:
        repositories = github_search_repositories(f'"{topic}" implementation', 10)

    if not repositories:
        st.info(
            "No closely matching public GitHub repositories were found for this topic. "
            "Unrelated repositories are not shown just to fill the list."
        )
        return

    for index, repo in enumerate(repositories[:3], start=1):
        name = normalize(repo.get("full_name")) or normalize(repo.get("name"))
        description = clean_text(repo.get("description"), 360)
        language = normalize(repo.get("language"))
        stars = repo.get("stargazers_count", 0)

        st.markdown(f"### {index}. {name}")
        if description:
            st.write(description)
        else:
            st.write("Public GitHub repository related to the selected topic.")

        metadata = []
        if language:
            metadata.append(f"Language: {language}")
        metadata.append(f"⭐ {stars:,}")
        st.caption(" • ".join(metadata))
        st.link_button("🔗 Open GitHub Project", repo["html_url"])
        st.divider()


def render_gemini_assistant(course, unit, topic, topic_options, research_items):
    st.subheader("🤖 Gemini AI Learning Assistant")
    st.caption(
        "Gemini explains the selected GPCET R23 CSE topic immediately "
        "using the official syllabus context. Live SerpApi resources are "
        "available separately below."
    )

    if not gemini_available():
        st.warning(
            "Gemini is not configured. Add GEMINI_API_KEY to your .env file."
        )
        return

    cache_key = f"gemini_answer::{course['code']}::{unit['unit']}::{topic}"

    if st.button(
        f"✨ Learn {topic} with Gemini",
        use_container_width=True,
        type="primary",
    ):
        prompt = build_gemini_prompt(
            course,
            unit,
            topic,
            research_items,
        )

        with st.spinner(
            f"Gemini is preparing a clear explanation of {topic}..."
        ):
            answer, error = call_gemini(prompt)

        if answer:
            st.session_state[cache_key] = answer
            st.session_state.pop(f"{cache_key}__error", None)

        elif error:
            st.session_state[f"{cache_key}__error"] = error

    if cache_key in st.session_state:
        st.markdown(st.session_state[cache_key])

        if st.button(
            "🔄 Regenerate Gemini Explanation",
            key=f"{cache_key}__regenerate",
        ):
            st.session_state.pop(cache_key, None)
            st.session_state.pop(f"{cache_key}__error", None)
            st.rerun()

    elif f"{cache_key}__error" in st.session_state:
        error = st.session_state[f"{cache_key}__error"]

        if "503" in error or "temporarily" in error.lower() or "overload" in error.lower():
            st.warning(
                "Gemini is temporarily busy. Theory2Tech automatically "
                "retried the request and tried fallback Flash models. "
                "Press the Gemini button again after a short wait."
            )
        else:
            st.error(f"Gemini request failed: {error}")

        st.info(
            "Your GPCET syllabus and the optional SerpApi resources remain "
            "available even when Gemini is temporarily unavailable."
        )

    # GitHub projects are based on Gemini's project reasoning, but the links
    # come from GitHub's public API so Theory2Tech never invents repositories.
    if cache_key in st.session_state:
        github_key = f"github::{course['code']}::{unit['unit']}::{topic}"
        if github_key not in st.session_state:
            with st.spinner("Finding 3 relevant GitHub projects..."):
                st.session_state[github_key] = extract_github_search_terms(
                    st.session_state[cache_key], topic
                )

        github_answer_key = f"{github_key}::answer"
        st.session_state[github_answer_key] = st.session_state[cache_key]
        render_github_projects(topic, st.session_state[github_answer_key])


def build_gemini_prompt(course, unit, topic, research_items):
    """Build the Gemini prompt with the complete prior GPCET learning path.

    For project ideas, Gemini receives every syllabus topic that occurs before
    the selected topic in the official GPCET R23 CSE sequence, including topics
    from earlier subjects, semesters, units, and the current unit. Gemini must
    choose only the previous topics that are actually relevant to the selected
    topic instead of using all of them blindly.
    """
    current_topics = unit.get("topics", [])

    # Build the complete curriculum in syllabus order.
    # The selected topic is the boundary: everything before it is already
    # learnable/previous knowledge; everything after it is future knowledge.
    all_prior_topics = []
    selected_found = False

    for semester in SYLLABUS.get("semesters", []):
        for prior_course in semester.get("courses", []):
            for prior_unit in prior_course.get("units", []):
                # Stop only when we reach the exact selected course + unit.
                is_selected_unit = (
                    prior_course.get("code") == course.get("code")
                    and prior_unit.get("unit") == unit.get("unit")
                )

                if is_selected_unit:
                    for prior_topic in prior_unit.get("topics", []):
                        if prior_topic == topic:
                            selected_found = True
                            break
                        all_prior_topics.append(
                            f"{prior_course.get('code')} — "
                            f"{prior_course.get('title')} — "
                            f"{prior_unit.get('unit')} {prior_unit.get('title')} — "
                            f"{prior_topic}"
                        )
                    break

                for prior_topic in prior_unit.get("topics", []):
                    all_prior_topics.append(
                        f"{prior_course.get('code')} — "
                        f"{prior_course.get('title')} — "
                        f"{prior_unit.get('unit')} {prior_unit.get('title')} — "
                        f"{prior_topic}"
                    )

                if selected_found:
                    break
            if selected_found:
                break
        if selected_found:
            break

    if all_prior_topics:
        previous_topics_text = "\n".join(
            f"- {item}" for item in all_prior_topics
        )
    else:
        previous_topics_text = "- No previous GPCET R23 CSE topic is available."

    research = []

    for item in research_items:
        title = normalize(item.get("title"))
        snippet = clean_text(
            item.get("snippet")
            or item.get("description")
            or "",
            700,
        )
        url = item.get("link") or item.get("url") or ""

        if title or snippet:
            research.append(
                f"TITLE: {title}\n"
                f"SUMMARY: {snippet}\n"
                f"URL: {url}"
            )

    research_text = "\n\n".join(research)

    if not research_text:
        research_text = (
            "No live web research was supplied. Use the official syllabus "
            "context and general technical knowledge. Do not invent citations "
            "or URLs."
        )

    return f"""
You are the teaching assistant inside Theory2Tech.

The student selected this exact topic from the official GPCET R23 CSE syllabus.

SUBJECT:
{course['code']} — {course['title']}

CURRENT UNIT:
{unit['unit']} — {unit['title']}

EXACT SELECTED TOPIC:
{topic}

CURRENT UNIT TOPICS:
{", ".join(current_topics)}

ALL PREVIOUS GPCET R23 CSE TOPICS:
{previous_topics_text}

LIVE WEB RESEARCH:
{research_text}

Your job is to explain ONLY the exact selected topic:
"{topic}"

Do not change the topic into a broader or different topic.
For example, if the topic is "History of Computers", explain the history
and evolution of computers, not the history of computer science in general.
If the topic is "AVL Trees", explain AVL Trees, not all balanced trees.

IMPORTANT FOR PROJECT IDEAS:
The section "What can you build with it?" must consider the student's
BROADER GPCET R23 CSE LEARNING HISTORY, not just the current unit or subject.

Use ALL PREVIOUS GPCET R23 CSE TOPICS as a knowledge pool. These topics can
come from earlier semesters, earlier subjects, earlier units, or earlier topics
within the current unit.

However, DO NOT use all previous topics automatically. First identify which
previous topics are technically relevant to the selected topic, then use only
those relevant concepts when designing projects.

The project ideas should demonstrate how concepts learned earlier can be
combined with the current topic to create something practical.

For each project:
1. The SELECTED TOPIC must play a meaningful role.
2. Reuse one or more relevant PREVIOUS GPCET TOPICS when appropriate.
3. Prefer natural technical combinations rather than random combinations.
4. Clearly state the relevant combination using:
   **Uses:** {topic} + [relevant previous topic(s)].
5. Make the projects realistic for a B.Tech CSE student.
6. Increase practical complexity across the 3 ideas when possible.

Example of the intended reasoning:
If the current topic is B-Tree, do not restrict project ideas to other
Advanced Data Structures topics. You may use earlier relevant concepts such
as programming, arrays, data structures, searching, complexity, or database-
related concepts if those topics appear before B-Tree in the supplied GPCET
syllabus. Choose only combinations that make technical sense.

VERY IMPORTANT:
- Never use a topic that appears AFTER the selected topic in the syllabus.
- Never claim the student already learned a future topic.
- Never force an unrelated previous topic into a project.
- The ALL PREVIOUS GPCET R23 CSE TOPICS list is a knowledge pool, not a list
  that must all be mentioned.
- Do not pretend these are existing GitHub repositories. They are project ideas.
- Your **Uses:** lines will be used by Theory2Tech to guide a real GitHub repository search.
- Make each **Uses:** line concise and technically meaningful so it can be used as a search hint.

Write the answer in the following exact structure.

### 📌 What is {topic}?
Give one clear definition in 1–2 sentences.
Use simple CSE-student language.

### 🎯 Why is it important?
Give exactly 3 useful points.
Each point must explain why the student should understand the topic.

### 🔍 Key Concepts
Give 4–6 important concepts/features directly related to {topic}.
For each concept, use:
**Concept:** short explanation.

Do not introduce unrelated syllabus topics.

### 🧩 Simple Example
Give one easy step-by-step example.
Use a small example, numbers, pseudocode, or a simple scenario when
appropriate to the topic.

### 🌍 Where is it used in the real world?
Give 2–4 concrete applications only when supportable.
For each:
**Application:** what the topic does there and why it is useful.

Do not claim that a technology is used in a specific company unless the
research explicitly supports it.

### 💻 What can you build with it?
Give exactly 3 realistic student project ideas.
Each project must:
1. Use the SELECTED TOPIC in a meaningful way.
2. Combine it with relevant topics from ALL PREVIOUS GPCET R23 CSE TOPICS
   when useful.
3. Be something a B.Tech CSE student could realistically implement.
4. Clearly state the combination:
   **Uses:** {topic} + [relevant previous topic(s)].
5. Increase in practical complexity across the 3 ideas when possible.

Do not pretend these are existing GitHub repositories.
These are project ideas only.
Do not use future syllabus topics as prerequisites.

### 📚 What should I learn next?
Select ONLY from the CURRENT UNIT TOPICS supplied above.
Use the topics immediately following "{topic}" in the current unit.
Do not invent topics and do not import topics from another syllabus.

### 📝 Quick takeaway
Give 2–3 sentences summarizing the most important thing the student
should remember.

COMPLETENESS REQUIREMENT:
Return ALL sections above in one response. Do not stop after the Simple Example.
The response MUST include the full "🌍 Where is it used in the real world?" section,
the full "💻 What can you build with it?" section, "📚 What should I learn next?",
and "📝 Quick takeaway" before ending.

IMPORTANT RULES:
- Be technically accurate.
- Respect the supplied GPCET R23 CSE syllabus order.
- Do not fabricate sources, URLs, companies, projects, applications,
  statistics, or historical facts.
- Do not cite a source that was not supplied in LIVE WEB RESEARCH.
- Treat project ideas as proposed ideas, not existing repositories.
- Do not repeat the live research verbatim.
- Do not mention that you are an AI.
- Do not use "..." as filler.
- Keep the language clear and suitable for a B.Tech CSE student.
- Prefer concrete explanations over generic statements.
- If the research is insufficient for a claim, say that the research
  is insufficient rather than guessing.
"""

def load_syllabus():
    if not os.path.exists(DATA_FILE):
        st.error(f"{DATA_FILE} was not found. Put it in the same folder as app.py.")
        st.stop()

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


SYLLABUS = load_syllabus()


def normalize(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def get_domain(url):
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def is_bad_url(url):
    domain = get_domain(url)
    return any(
        domain == bad or domain.endswith("." + bad)
        for bad in BAD_DOMAINS
    )


def get_client():
    api_key = os.getenv("SERPAPI_KEY")

    if not api_key:
        st.error("SERPAPI_KEY is missing. Add it to your .env file.")
        st.stop()

    return serpapi.Client(api_key=api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def google_search(query, num=10):
    # A cached function cannot safely mutate session state, so the actual
    # guarded request is handled by the helper below.
    return _google_search_uncached(query, num)


def _google_search_uncached(query, num=10):
    if st.session_state.get("serpapi_unavailable", False):
        return []

    try:
        client = get_client()

        result = client.search({
            "engine": "google",
            "q": query,
            "hl": "en",
            "gl": "in",
            "num": min(int(num), 10),
            "safe": "active",
        })

        return result.get("organic_results", [])

    except Exception as exc:
        st.session_state.serpapi_unavailable = True
        st.session_state.serpapi_error_message = (
            "SerpApi YouTube search could not be reached from this computer. "
            f"{type(exc).__name__}: {exc}"
        )
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def youtube_search(query, num=10):
    return _youtube_search_uncached(query, num)


def _youtube_search_uncached(query, num=10):
    if st.session_state.get("serpapi_unavailable", False):
        return []

    try:
        client = get_client()

        result = client.search({
            "engine": "youtube",
            "search_query": query,
            "hl": "en",
            "gl": "in",
            "num": min(int(num), 10),
            "safe": "active",
        })

        return result.get("video_results", [])

    except Exception as exc:
        st.session_state.serpapi_unavailable = True
        st.session_state.serpapi_error_message = (
            "SerpApi YouTube search could not be reached from this computer. "
            f"{type(exc).__name__}: {exc}"
        )
        return []


def item_text(item):
    return normalize(
        " ".join([
            str(item.get("title", "")),
            str(item.get("snippet", "")),
            str(item.get("description", "")),
            str(item.get("source", "")),
        ])
    )


def topic_variants(topic):
    topic = normalize(topic)
    variants = [topic]

    if re.search(r"\bb[- ]?tree(s)?\b", topic, re.I):
        variants += ["B-Tree", "B Tree", "B-Trees", "B Trees"]

    if re.search(r"\bb\+[- ]?tree(s)?\b", topic, re.I):
        variants += ["B+ Tree", "B Plus Tree"]

    return list(dict.fromkeys(variants))


def topic_matches(text, topic, strict=False):
    text = normalize(text).lower()
    variants = [v.lower() for v in topic_variants(topic)]

    if any(v in text for v in variants):
        return True

    words = [
        w.lower()
        for w in re.findall(r"[a-z0-9]+", topic)
        if len(w) >= 3
    ]

    if not words:
        return True

    matches = sum(w in text for w in words)
    threshold = len(words) if strict else max(1, len(words) - 1)

    return matches >= threshold


def has_syllabus_noise(text):
    low = normalize(text).lower()
    hits = sum(1 for word in SYLLABUS_NOISE if word in low)
    return hits >= 2


def clean_text(text, max_length=420):
    text = normalize(text)
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = text.replace("...", "")
    text = normalize(text)

    if len(text) <= max_length:
        return text

    return text[:max_length].rsplit(" ", 1)[0]


def deduplicate(results, limit=5):
    output = []
    seen = set()

    for item in results:
        url = item.get("link") or item.get("url")

        if not url or is_bad_url(url):
            continue

        key = url.rstrip("/").lower()

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

        if len(output) >= limit:
            break

    return output


def domain_score(url, domain_scores):
    return domain_scores.get(get_domain(url), 0)


def rank_results(results, topic, category="general"):
    ranked = []

    for item in results:
        url = item.get("link") or item.get("url", "")
        text = item_text(item)

        if not url or is_bad_url(url):
            continue

        if not topic_matches(text, topic):
            continue

        score = 0
        low = text.lower()

        if any(v.lower() in low for v in topic_variants(topic)):
            score += 8

        if has_syllabus_noise(text):
            score -= 8

        if category == "definition":
            score += domain_score(url, TRUSTED_DEFINITION_DOMAINS)
            if "homework" in low or "assignment" in low:
                score -= 8
            if low.startswith("question:"):
                score -= 8
            score += sum(
                word in low
                for word in [
                    "is a",
                    "is an",
                    "refers to",
                    "defined as",
                    "data structure",
                    "tree data structure",
                ]
            ) * 2

        elif category == "learning":
            score += domain_score(url, TRUSTED_LEARNING_DOMAINS)
            score += sum(word in low for word in EDUCATION_WORDS)

        elif category == "course":
            score += domain_score(url, COURSE_DOMAINS)
            score += sum(word in low for word in COURSE_WORDS) * 2

            if get_domain(url) not in COURSE_DOMAINS:
                score -= 3

        elif category == "application":
            score += sum(word in low for word in APPLICATION_WORDS) * 2
            score -= sum(word in low for word in COURSE_WORDS) * 2
            score -= sum(
                word in low
                for word in [
                    "syllabus",
                    "curriculum",
                    "question bank",
                    "course outcome",
                    "exam",
                ]
            ) * 4

        elif category == "project":
            score += sum(word in low for word in PROJECT_WORDS)

            if "github.com/" in url.lower():
                score += 8

        ranked.append((score, item))

    ranked.sort(key=lambda pair: pair[0], reverse=True)

    return deduplicate(
        [item for _, item in ranked],
        5,
    )


def clean_definition(text):
    text = clean_text(text, 360)

    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        sentence = normalize(sentence)

        if 35 <= len(sentence) <= 300:
            return sentence.rstrip(".!?") + "."

    return text.rstrip(".!?") + "."


def search_definition(topic):
    queries = [
        f'"{topic}" definition',
        f'"{topic}" computer history',
    ]
    results = []
    for query in queries:
        results.extend(google_search(query, 10))
    return rank_results(results, topic, "definition")

def search_notes(topic):
    queries = [
        f'"{topic}" notes tutorial',
        f'"{topic}" lecture notes',
    ]
    results = []
    for query in queries:
        results.extend(google_search(query, 10))

    ranked = rank_results(results, topic, "learning")

    return [
        item
        for item in ranked
        if get_domain(item.get("link", "")) not in {
            "youtube.com",
            "youtu.be",
        }
    ][:5]

def search_youtube(topic):
    """
    SerpApi is used ONLY for YouTube resources in Theory2Tech.

    One selected topic = one YouTube search.
    """
    query = f'"{topic}" tutorial explained'
    results = youtube_search(query, 10)

    output = []
    seen = set()

    for item in results:
        link = item.get("link", "")

        if "youtube.com/watch" not in link.lower():
            continue

        key = link.rstrip("/").lower()

        if not key or key in seen:
            continue

        seen.add(key)
        output.append(item)

        if len(output) >= 3:
            break

    return output


def search_courses(topic):
    queries = [
        f'"{topic}" free course certificate'
    ]

    results = []

    for query in queries:
        results.extend(google_search(query, 10))

    ranked = rank_results(results, topic, "course")

    course_results = [
        item
        for item in ranked
        if get_domain(item.get("link", "")) in COURSE_DOMAINS
        and not has_syllabus_noise(item_text(item))
    ]

    return course_results[:5]


def search_real_world(topic):
    queries = [
        f'"{topic}" real world applications computer science',
        f'"{topic}" practical applications technology',
        f'"{topic}" systems applications implementation',
    ]
    results = []
    for query in queries:
        results.extend(google_search(query, 10))
    return rank_results(results, topic, "application")

def is_real_github_repo(url):
    if "github.com/" not in url.lower():
        return False

    path = urlparse(url).path.strip("/").split("/")

    if len(path) != 2:
        return False

    blocked = {
        "topics",
        "search",
        "collections",
        "marketplace",
        "features",
        "trending",
        "orgs",
    }

    owner, repo = path[0].lower(), path[1].lower()

    if owner in blocked or repo in blocked:
        return False

    return True


def search_projects(topic):
    queries = [
        f'site:github.com "{topic}" implementation',
        f'site:github.com "{topic}" project',
        f'site:github.com "{topic}" visualization',
    ]
    results = []
    for query in queries:
        results.extend(google_search(query, 10))

    candidates = []
    for item in results:
        url = item.get("link", "")
        if not is_real_github_repo(url):
            continue
        text = item_text(item).lower()
        if not topic_matches(text, topic):
            continue
        if not any(
            word in text
            for word in ["project", "implementation", "simulator", "visualization", "timeline"]
        ):
            continue
        candidates.append(item)

    return rank_results(candidates, topic, "project")

def show_result_card(item, button_text, description_fallback):
    title = normalize(item.get("title")) or "Resource"

    st.markdown(f"**{title}**")

    description = clean_text(
        item.get("snippet")
        or item.get("description")
        or description_fallback,
        420,
    )

    if description:
        st.write(description)

    url = item.get("link") or item.get("url")

    if url:
        st.link_button(button_text, url)

    source = item.get("source") or get_domain(url)

    if source:
        st.caption(f"Source: {source}")


def render_definition_from_results(topic, results):
    st.subheader("1️⃣ What Is It?")
    st.caption(f"Live research topic: **{topic}**")

    if not results:
        st.info("No reliable live definition was found for this exact topic.")
        return

    result = results[0]
    raw = (
        result.get("snippet")
        or result.get("description")
        or result.get("title")
    )

    definition = clean_definition(raw)
    st.markdown(f"**{definition}**")

    source = result.get("source") or get_domain(result.get("link", ""))
    st.caption(f"Source: {source}")

    if result.get("link"):
        st.link_button("Open Definition Source", result["link"])


def render_learning_resources_from_results(topic, notes, youtube, courses):
    st.subheader("2️⃣ Learning Resources")

    st.markdown("### 📄 Notes & Tutorials")
    if notes:
        for item in notes:
            show_result_card(
                item,
                "Open Resource",
                "Tutorial or notes explaining the selected syllabus topic.",
            )
            st.divider()
    else:
        st.info("No closely matching notes or tutorials were found.")

    st.markdown("### ▶️ YouTube Resources")
    if youtube:
        for item in youtube:
            title = normalize(item.get("title")) or "YouTube Resource"
            st.markdown(f"**{title}**")

            description = clean_text(item.get("description"), 350)
            if description:
                st.write(description)

            channel = item.get("channel")
            if isinstance(channel, dict):
                channel_name = normalize(channel.get("name"))
                if channel_name:
                    st.caption(f"Channel: {channel_name}")

            if item.get("link"):
                st.link_button("Watch on YouTube", item["link"])

            st.divider()
    else:
        st.info("No closely matching YouTube resources were found.")

    st.markdown("### 🎓 Free Courses / Certificates")
    if courses:
        for item in courses:
            show_result_card(
                item,
                "View Course",
                "Course or learning page related to the selected topic.",
            )
            st.divider()

        st.caption(
            "Certificate availability, fees and eligibility can change. "
            "Verify the provider's current terms before enrolling."
        )
    else:
        st.info("No reliable free course or certificate page was found.")


def render_real_world_from_results(topic, results):
    st.subheader("3️⃣ Where Is It Used in the Real World?")

    if not results:
        st.info(
            "No strong topic-specific real-world application source was found. "
            "Generic computer uses are not substituted."
        )
        return

    for item in results:
        show_result_card(
            item,
            "Read Application Details",
            "Application-focused information about the selected topic.",
        )
        st.divider()


def render_projects_from_results(topic, results):
    st.subheader("4️⃣ What Projects Can You Build?")

    if not results:
        st.info(
            "No closely matching GitHub project was found for this exact topic. "
            "Unrelated repositories are not shown just to fill the list."
        )
        return

    for item in results:
        show_result_card(
            item,
            "Open GitHub Project",
            "GitHub repository containing a project or implementation of the selected topic.",
        )
        st.divider()


def render_definition(topic):
    st.subheader("1️⃣ What Is It?")
    st.caption(f"Live research topic: **{topic}**")

    results = search_definition(topic)

    if not results:
        st.info("No reliable live definition was found for this exact topic.")
        return results

    result = results[0]

    raw = (
        result.get("snippet")
        or result.get("description")
        or result.get("title")
    )

    definition = clean_definition(raw)

    st.markdown(f"**{definition}**")

    source = result.get("source") or get_domain(result.get("link", ""))
    st.caption(f"Source: {source}")

    if result.get("link"):
        st.link_button(
            "Open Definition Source",
            result["link"],
        )

    return results


def render_learning_resources(topic):
    st.subheader("2️⃣ Learning Resources")

    notes = search_notes(topic)

    st.markdown("### 📄 Notes & Tutorials")

    if notes:
        for item in notes:
            show_result_card(
                item,
                "Open Resource",
                "Tutorial or notes explaining the selected syllabus topic.",
            )
            st.divider()
    else:
        st.info("No closely matching notes or tutorials were found.")

    youtube = search_youtube(topic)

    st.markdown("### ▶️ YouTube Resources")

    if youtube:
        for item in youtube:
            title = normalize(item.get("title")) or "YouTube Resource"

            st.markdown(f"**{title}**")

            description = clean_text(
                item.get("description"),
                350,
            )

            if description:
                st.write(description)

            channel = item.get("channel")

            if isinstance(channel, dict):
                channel_name = normalize(channel.get("name"))

                if channel_name:
                    st.caption(f"Channel: {channel_name}")

            if item.get("link"):
                st.link_button(
                    "Watch on YouTube",
                    item["link"],
                )

            st.divider()
    else:
        st.info("No closely matching YouTube resources were found.")

    courses = search_courses(topic)

    st.markdown("### 🎓 Free Courses / Certificates")

    if courses:
        for item in courses:
            show_result_card(
                item,
                "View Course",
                "Course or learning page related to the selected topic.",
            )
            st.divider()

        st.caption(
            "Certificate availability, fees and eligibility can change. "
            "Verify the provider's current terms before enrolling."
        )
    else:
        st.info("No reliable free course or certificate page was found.")

    return notes, youtube, courses


def render_real_world(topic):
    st.subheader("3️⃣ Where Is It Used in the Real World?")

    results = search_real_world(topic)

    if not results:
        st.info("No strong topic-specific real-world application source was found. Generic computer uses are not substituted.")
        return results

    for item in results:
        show_result_card(
            item,
            "Read Application Details",
            "Application-focused information about the selected topic.",
        )
        st.divider()

    return results


def render_projects(topic):
    st.subheader("4️⃣ What Projects Can You Build?")

    results = search_projects(topic)

    if not results:
        st.info("No closely matching GitHub project was found for this exact topic. Unrelated repositories are not shown just to fill the list.")
        return results

    for item in results:
        show_result_card(
            item,
            "Open GitHub Project",
            "GitHub repository containing a project or implementation of the selected topic.",
        )
        st.divider()

    return results


def main():
    st.sidebar.title("📚 GPCET R23 CSE")

    st.sidebar.caption(
        "Core CSE theory subjects only. Labs, electives and non-CSE courses are removed."
    )

    semester_options = [
        f"{semester['year']} • {semester['semester']}"
        for semester in SYLLABUS["semesters"]
    ]

    selected_semester_label = st.sidebar.selectbox(
        "1. Select Semester",
        semester_options,
    )

    semester = SYLLABUS["semesters"][
        semester_options.index(selected_semester_label)
    ]

    course_options = [
        f"{course['code']} — {course['title']}"
        for course in semester["courses"]
    ]

    selected_course_label = st.sidebar.selectbox(
        "2. Select Core CSE Subject",
        course_options,
    )

    course = semester["courses"][
        course_options.index(selected_course_label)
    ]

    units = course.get("units", [])

    st.title("🌍 Theory2Tech")
    st.caption("From GPCET R23 CSE Syllabus to Real-World Technology")

    if not units:
        st.info(
            f"**{course['title']}** is present in the GPCET curriculum, "
            "but detailed unit topics were not recoverable from the uploaded PDF."
        )

        if course.get("source_note"):
            st.warning(course["source_note"])

        st.stop()

    unit_options = [
        f"{unit['unit']} — {unit['title']}"
        for unit in units
    ]

    selected_unit_label = st.sidebar.selectbox(
        "3. Select Current Unit",
        unit_options,
    )

    unit_index = unit_options.index(selected_unit_label)
    unit = units[unit_index]

    topic_options = unit.get("topics", [])

    selected_topic = st.sidebar.selectbox(
        "4. Select Topic",
        topic_options,
    )

    topic_index = topic_options.index(selected_topic)

    st.markdown(f"### {course['title']}")

    st.markdown(
        f"**{course['code']}** • "
        f"{course['year']} • "
        f"{course['semester']} • "
        f"**{unit['unit']} — {unit['title']}**"
    )

    if course.get("source_note"):
        with st.expander("ℹ️ PDF extraction/source note"):
            st.write(course["source_note"])

    with st.expander(
        f"📖 Topics in {unit['unit']} — {unit['title']}",
        expanded=True,
    ):
        for index, topic in enumerate(topic_options, start=1):
            if topic == selected_topic:
                st.markdown(
                    f"**{index}. {topic} ← Selected**"
                )
            else:
                st.markdown(f"{index}. {topic}")

    st.divider()

    if st.session_state.get("serpapi_unavailable", False):
        st.error(
            "Live SerpApi research is temporarily unavailable from this computer. "
            "The syllabus and previously cached results can still be displayed. "
            "Restart Streamlit after checking your internet/firewall connection."
        )

        if st.session_state.get("serpapi_error_message"):
            with st.expander("Technical error details"):
                st.code(st.session_state.serpapi_error_message)

    # ---------------------------------------------------------
    # PRIMARY LEARNING EXPERIENCE
    # ---------------------------------------------------------
    # Gemini must NOT wait for SerpApi. Previously the app performed
    # several web searches first, which could leave the page stuck at
    # "Collecting live SerpApi research for Gemini...".
    #
    # Gemini now works independently from the official syllabus context.
    # SerpApi is an optional supporting layer below.
    # ---------------------------------------------------------

    render_gemini_assistant(
        course,
        unit,
        selected_topic,
        topic_options,
        [],
    )

    st.divider()

    # ---------------------------------------------------------
    # SERPAPI — YOUTUBE ONLY
    # ---------------------------------------------------------
    # Gemini already provides the definition, importance, key concepts,
    # examples, real-world usage, project ideas and next topics.
    #
    # SerpApi is therefore used ONLY for YouTube learning resources.
    # One topic triggers only one YouTube search.
    # Show only the first 3 relevant results to keep the UI focused.
    # ---------------------------------------------------------

    st.divider()
    st.subheader("▶️ YouTube Learning Resources")
    st.caption(
        "SerpApi is used only to find YouTube videos for the selected "
        "GPCET topic. One topic uses one YouTube search. Showing up to 3 resources."
    )

    youtube_key = (
        f"youtube::{course['code']}::{unit['unit']}::{selected_topic}"
    )

    if st.button(
        f"🔎 Find YouTube Resources for {selected_topic}",
        key=f"{youtube_key}::load",
        use_container_width=True,
    ):
        with st.spinner("Searching YouTube with SerpApi..."):
            youtube_results = search_youtube(selected_topic)

        st.session_state[youtube_key] = youtube_results

    youtube_results = st.session_state.get(youtube_key)

    if youtube_results is None:
        st.info(
            "Gemini provides the main explanation. Click the button above "
            "when you want YouTube learning resources."
        )
    elif youtube_results:
        for item in youtube_results:
            title = normalize(item.get("title")) or "YouTube Resource"
            st.markdown(f"**{title}**")

            description = clean_text(
                item.get("description"),
                350,
            )

            if description:
                st.write(description)

            channel = item.get("channel")

            if isinstance(channel, dict):
                channel_name = normalize(channel.get("name"))
                if channel_name:
                    st.caption(f"Channel: {channel_name}")

            if item.get("link"):
                st.link_button(
                    "▶️ Watch on YouTube",
                    item["link"],
                )

            st.divider()
    else:
        st.info(
            "No closely matching YouTube resources were found for this topic."
        )

    if st.session_state.get("serpapi_unavailable", False):
        st.warning(
            "SerpApi YouTube search is temporarily unavailable. "
            "The Gemini learning assistant remains available."
        )

        if st.session_state.get("serpapi_error_message"):
            with st.expander("Technical details"):
                st.code(st.session_state.serpapi_error_message)

        if st.button("🔄 Retry YouTube Search"):
            st.session_state.serpapi_unavailable = False
            st.session_state.serpapi_error_message = ""
            st.cache_data.clear()
            st.rerun()

    st.caption(
        "Theory2Tech uses the GPCET R23 CSE syllabus for the learning path, "
        "Gemini for the student-friendly explanation and project direction, "
        "GitHub for real repository links, and SerpApi only for YouTube learning resources. "
        "No OpenAI API is used."
    )


if __name__ == "__main__":
    main()
