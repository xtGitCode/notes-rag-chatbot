from fastapi import FastAPI
from app.routers import notes, chat  # Correct absolute import
from app.llm.llm_handler import LlamaHandler
from app.config import HF_TOKEN, MODEL_NAME

app = FastAPI(title="Notes RAG Chatbot API")

# Initialize the Llama handler ONCE at startup
try:
    app.llama_handler = LlamaHandler(MODEL_NAME, HF_TOKEN)
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit(1) #Exit if LLM fails to load

app.include_router(notes.router, prefix="/notes", tags=["Notes"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Notes RAG Chatbot API"}