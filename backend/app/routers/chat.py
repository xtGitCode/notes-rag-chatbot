from fastapi import APIRouter, HTTPException
from backend.app.models import ChatRequest, ChatResponse
from ..llm.llm_handler import NotesRAGBot

router = APIRouter()
bot = NotesRAGBot()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        answer = bot.query(chat_request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))