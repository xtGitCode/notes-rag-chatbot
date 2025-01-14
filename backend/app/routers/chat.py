from fastapi import APIRouter
from app.models import ChatMessage
from app.database import query_chroma

router = APIRouter()

@router.post("/chat")
async def chat(message: ChatMessage):
    results = query_chroma(message.query)
    return {"results": results}