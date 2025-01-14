from fastapi import APIRouter
from backend.app.models import ChatMessage
from backend.app.database import query_chroma

router = APIRouter()

@router.post("/chat")
async def chat(message: ChatMessage):
    results = query_chroma(message.query)
    return {"results": results}