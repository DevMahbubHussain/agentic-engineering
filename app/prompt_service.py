import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", api_key=GOOGLE_API_KEY)


def generate_single_prompt(topic: str, tone: str, length: str) -> dict:
    """Same logic as your CLI single_prompt_demo(), but takes args and returns a dict
    instead of calling input()/print()."""
    template_text = (
        "Write a {tone} explanation about the topic: {topic}. "
        "Make the explanation {length} and easy to understand"
    )
    prompt = PromptTemplate(
        input_variables=["topic", "tone", "length"], template=template_text
    )
    final_prompt = prompt.format(topic=topic, tone=tone, length=length)

    response = llm.invoke(final_prompt)

    return {"generated_prompt": final_prompt, "response": response.content}


def generate_chat_prompt(system_role: str, user_question: str, tone: str) -> dict:
    """Same logic as your CLI chat_prompt_demo(), fixed to end on a user turn
    (tone folded into the system message instead of a fake assistant turn)."""
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_role} Respond in a {tone} tone."),
            ("user", "{user_question}"),
        ]
    )
    formatted_chat = chat_template.format_messages(
        system_role=system_role, user_question=user_question, tone=tone
    )

    response = llm.invoke(formatted_chat)

    messages_preview = [
        {"role": msg.type, "content": msg.content} for msg in formatted_chat
    ]

    return {"messages": messages_preview, "response": response.content}
