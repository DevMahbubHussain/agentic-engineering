import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables once, when this module is first imported
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Tried in order until one actually works — keeps you running even if
# Google renames/deprecates a model or one is temporarily unavailable.
CANDIDATE_MODELS = [
    os.getenv("GEMINI_MODEL"),  # optional override via .env
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
]
CANDIDATE_MODELS = [m for m in CANDIDATE_MODELS if m]

_cached_llm = None


def get_api_key() -> str:
    if not GOOGLE_API_KEY:
        raise ValueError("No API key found. Set GOOGLE_API_KEY in your .env")
    return GOOGLE_API_KEY


def pick_working_model(api_key: str) -> str:
    """Try each candidate model with a cheap real call; return the first that works."""
    last_error = None
    for name in CANDIDATE_MODELS:
        try:
            test_llm = ChatGoogleGenerativeAI(
                model=name, api_key=api_key, temperature=0
            )
            test_llm.invoke("ping")
            print(f"[model check] Using '{name}'\n")
            return name
        except Exception as e:
            print(
                f"[model check] '{name}' unavailable ({type(e).__name__}), trying next..."
            )
            last_error = e
    raise RuntimeError(f"None of the candidate models worked. Last error: {last_error}")


def get_llm(temperature: float = 0.4, force_refresh: bool = False):
    """Returns a ready-to-use ChatGoogleGenerativeAI instance.
    Caches the working model so repeated calls skip re-testing candidates."""
    global _cached_llm

    if _cached_llm is not None and not force_refresh:
        return _cached_llm

    api_key = get_api_key()
    model_name = pick_working_model(api_key)
    _cached_llm = ChatGoogleGenerativeAI(
        model=model_name, temperature=temperature, api_key=api_key
    )
    return _cached_llm
