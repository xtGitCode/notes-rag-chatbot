from fastapi import FastAPI
from backend.app.routers import notes, chat
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")

app = FastAPI(title="Notes RAG Chatbot API")

app.include_router(notes.router, prefix="/notes", tags=["Notes"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Notes RAG Chatbot API"}