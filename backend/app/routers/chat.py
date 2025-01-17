from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..llm.llm_handler import NotesRAGBot

router = APIRouter()
bot = NotesRAGBot()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        raw_answer = bot.query(chat_request.question)  # Query the bot
        if isinstance(raw_answer, dict):  # Handle dictionary responses
            answer = raw_answer.get("answer", "No valid answer provided.")
        elif isinstance(raw_answer, str):  # Ensure it's a string
            answer = raw_answer
        else:
            raise ValueError("Unexpected response type from bot.")
        return {"answer": answer}  # Return a response that matches the schema
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
