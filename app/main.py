from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from prompt_service import generate_single_prompt, generate_chat_prompt

load_dotenv()

app = FastAPI(title="Gemini Prompt Engineering API")
# templates = Jinja2Templates(directory="templates")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/api/prompt-template")
def prompt_template_endpoint(
    topic: str = Form(...), tone: str = Form(...), length: str = Form(...)
):
    """Equivalent of your CLI option 1 (PromptTemplate workflow)."""
    return generate_single_prompt(topic, tone, length)


@app.post("/api/chat-template")
def chat_template_endpoint(
    system_role: str = Form(...),
    user_question: str = Form(...),
    tone: str = Form(...),
):
    """Equivalent of your CLI option 2 (ChatPromptTemplate workflow)."""
    return generate_chat_prompt(system_role, user_question, tone)


# Run with: uvicorn main:app --reload
