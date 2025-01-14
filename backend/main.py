from fastapi import FastAPI
from app.routers import notes, chat

app = FastAPI(title="Notes RAG Chatbot API")

app.include_router(notes.router, prefix="/notes", tags=["Notes"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Notes RAG Chatbot API"}