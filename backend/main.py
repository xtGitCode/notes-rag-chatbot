from fastapi import FastAPI
from fastapi.responses import JSONResponse
from backend.app.routers import notes, chat
from dotenv import load_dotenv
import uuid

load_dotenv()
app = FastAPI(title="Notes RAG Chatbot API")

app.include_router(notes.router, prefix="/notes", tags=["Notes"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Notes RAG Chatbot API"}