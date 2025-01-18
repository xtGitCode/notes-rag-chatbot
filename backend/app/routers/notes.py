from fastapi import APIRouter, HTTPException, Depends
from backend.app.models import NoteCreate, Note
from backend.app.database import add_note_to_pinecone, delete_note_from_pinecone, index, namespace
from typing import List
from fastapi import APIRouter, HTTPException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/notes", response_model=Note, status_code=201)
def create_note(note: NoteCreate):
    """Create a new note and add it to Pinecone."""
    try:
        created_note = add_note_to_pinecone(note.title, note.content)
        if created_note:
            return created_note
        else:
            raise HTTPException(status_code=500, detail="Failed to create note")
    except Exception as e:
        logger.error(f"Error creating note: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.get("/notes", response_model=List[Note])
async def get_all_notes():
    """Retrieve all notes from Pinecone."""
    try:
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        
        if total_vectors == 0:
            return []
        
        query_response = index.query(
            vector=[0] * 768,
            top_k=total_vectors,
            include_metadata=True,
            namespace=namespace
        )
        
        all_notes = []
        for match in query_response.matches:
            note = Note(
                id=match.id,
                title=match.metadata.get("title", "Untitled"),
                content=match.metadata.get("content", "")  # Now content will be retrieved from metadata
            )
            all_notes.append(note)
            
        logging.info(f"Successfully retrieved {len(all_notes)} notes")
        return all_notes
        
    except Exception as e:
        logging.error(f"Error retrieving notes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while retrieving notes: {str(e)}")

@router.delete("/notes/{note_id}", status_code=204)
def delete_note(note_id: str):
    """Delete a note from Pinecone using its ID."""
    try:
        if not delete_note_from_pinecone(note_id):
            raise HTTPException(status_code=404, detail="Note not found")
    except Exception as e:
        logger.error(f"Error deleting note: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete note")