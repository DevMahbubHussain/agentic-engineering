from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

import os

# Tried in order until one actually responds. Keeps you running even if
# Google deprecates/restricts one of these for your project.
CANDIDATE_MODELS = [
    os.getenv("GEMINI_MODEL"),  # explicit override, if set in .env
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
]
CANDIDATE_MODELS = [m for m in CANDIDATE_MODELS if m]  # drop None if no override


def extract_text(content) -> str:
    """response.content can be a str or a list of content blocks
    (e.g. with Gemini 3.x thinking models). Normalize to plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Common shapes: {"type": "text", "text": "..."}
                text = block.get("text")
                if text:
                    parts.append(text)
        return "\n".join(parts)

    return str(content)


def pick_working_model(api_key: str) -> str:
    """Try each candidate with a cheap real call; return the first that works."""
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


def build_chain():
    # Load environment variables from .env
    load_dotenv()

    # The LangChain integration checks GOOGLE_API_KEY first, then GEMINI_API_KEY
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env"
        )

    model_name = pick_working_model(api_key)

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.4,
        api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a senior software architect helping developers make "
                    "good technical decisions. Be concise, practical, and specific. "
                    "Focus on architecture, tools, trade-offs, and best practices."
                ),
            ),
            (
                "human",
                (
                    "Developer question:\n"
                    "{question}\n\n"
                    "Answer for an experienced tech audience. "
                    "Use short paragraphs and bullets when helpful."
                ),
            ),
        ]
    )

    # LangChain Expression Language: prompt -> model
    chain = prompt | llm
    return chain, model_name


def main():
    chain, model_name = build_chain()

    print(f"Tech Stack Advisor (Gemini via LangChain, model={model_name})")
    print("Ask architecture / tooling questions (type 'exit' to quit).\n")

    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye")
            break

        if user_input.strip().lower() in {"exit", "quit", "q"}:
            print("Goodbye")
            break

        if not user_input.strip():
            continue

        try:
            response = chain.invoke({"question": user_input})
            print("\nAI:\n" + extract_text(response.content) + "\n")
        except Exception as e:
            print(f"Error while calling the model: {e}\n")


if __name__ == "__main__":
    main()