from pydantic import BaseModel

class NoteCreate(BaseModel):
    title: str
    content: str

class Note(BaseModel):
    id: str
    title: str
    content: str

class ChatMessage(BaseModel):
    query: str