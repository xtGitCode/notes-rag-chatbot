from fastapi import APIRouter, HTTPException
from backend.app.models import NoteCreate, Note
from backend.app.database import add_note_to_chroma, client, collection, delete_note_from_chroma
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/notes", response_model=Note, status_code=201)
def create_note(note: NoteCreate):
    try:
        created_note = add_note_to_chroma(note.title, note.content)
        if created_note:
            return {"id": "generated_uuid", "title": note.title, "content": note.content}  
        else:
            raise HTTPException(status_code=500, detail="Failed to create note")
    except Exception as e:
        logger.error(f"Error creating note: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
@router.get("/notes", response_model=List[Note])
def get_all_notes():
    try:
        # Fetch all documents from the collection
        all_notes = collection.get()
        
        # Extract documents, ids, and metadata
        documents = all_notes.get("documents", [])
        ids = all_notes.get("ids", [])
        metadatas = all_notes.get("metadatas", [])
        
        # Combine the data into Note objects
        return [
            Note(id=id_, title=metadata.get("title", ""), content=document)
            for id_, metadata, document in zip(ids, metadatas, documents)
        ]
    except Exception as e:
        logger.error(f"Error retrieving notes: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch notes")

@router.delete("/notes/{note_id}", status_code=204)
def delete_note(note_id: str):
    try:
        if not delete_note_from_chroma(note_id):
            raise HTTPException(status_code=404, detail="Note not found")
    except Exception as e:
        print(f"Error deleting from Chroma: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete note")

@router.get("/chroma/count")
def get_chroma_count():
    return {"count": collection.count()}