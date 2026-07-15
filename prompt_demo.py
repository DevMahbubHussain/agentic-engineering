import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# from langchain_google_genai import ChatGoogleGenerativeAI
from gemini_model import get_llm

# Load environment variables from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini LLM wrapper (LangChain)
# llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", api_key=GOOGLE_API_KEY)


# Single Prompt Demo
def single_prompt_demo():
    print("\n--- PromptTemplate ---")
    # Ask user input
    topic = input("Topic: ")
    tone = input("Tone: ")
    length = input("Choose length (short / medium / long): ")

    # Template with  3 variable
    template_text = (
        "Write a {tone} explanation about the topic: {topic}. "
        "Make the explanation {length} and easy to understand"
    )

    # Create PromptTemplate

    prompt = PromptTemplate(
        input_variables=["topic", "tone", "length"], template=template_text
    )

    final_prompt = prompt.format(topic=topic, tone=tone, length=length)
    print("\nGenerated Prompt:")
    print(final_prompt)

    llm = get_llm()
    print("\nSending to Gemini...\n")
    response = llm.invoke(final_prompt)

    print("Gemini Response:")
    print(response.content)


def chat_prompt_demo():
    print("\n--- ChatPromptTemplate ---")

    system_role = input("Enter system instruction (e.g., You are a tutor): ")
    user_question = input("Enter user message: ")
    tone = input("Choose AI tone (friendly / strict / expert): ")

    # Chat template with structured message roles
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_role}"),
            ("user", "{user_question}"),
        ]
    )

    # Format chat messages
    formatted_chat = chat_template.format_messages(
        system_role=system_role, user_question=user_question, tone=tone
    )

    print("\nGenerated Chat Prompt:")
    for msg in formatted_chat:
        print(f"{msg.type.upper()}: {msg.content}")

    llm = get_llm()
    print("\nSending to Gemini...\n")
    response = llm.invoke(formatted_chat)

    print("Gemini Reply:")
    print(response.content)
