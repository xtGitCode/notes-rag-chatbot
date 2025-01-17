from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm.llm_handler import NotesRAGBot
from backend.app.models import ChatRequest, ChatResponse

router = APIRouter()
bot = NotesRAGBot()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        raw_answer = bot.query(chat_request.question)  # Query the bot

        if isinstance(raw_answer, dict):  # Handle dictionary responses
            answer = raw_answer.get("answer", "No valid answer provided.")
            title = raw_answer.get("title", "No Title Found")  # Set default title
            content = raw_answer.get("content", "No Content Found")  # Set default content
        elif isinstance(raw_answer, str):  # Ensure it's a string
            answer = raw_answer
            title = "No Title Found"  # Set default title (optional)
            content = "No Content Found"  # Set default content (optional)
        else:
            raise ValueError("Unexpected response type from bot.")

        return ChatResponse(answer=answer, title=title, content=content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
