from fastapi import APIRouter, HTTPException
from ..llm.llm_handler import NotesRAGBot
from backend.app.models import ChatRequest, ChatResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()
bot = NotesRAGBot()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Chat endpoint that processes user questions and returns answers from the RAG system.
    """
    try:
        logger.info(f"Received chat request: {chat_request.question}")

        # Query bot
        response = bot.query(question=chat_request.question)

        # Process the response
        if isinstance(response, dict):
            answer = response.get("answer", "No valid answer provided.")
            title = response.get("title")
            content = response.get("content")

            logger.info(f"Generated response: {answer}")

            return ChatResponse(
                answer=answer,
                title=title,
                content=content
            )
        else:
            logger.error("Unexpected response type from bot.")
            raise ValueError("Unexpected response type from bot.")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
