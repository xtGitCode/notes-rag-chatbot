from fastapi import APIRouter, HTTPException
from app.models import NoteCreate, NoteUpdate

router = APIRouter()

notes_db = {}

@router.post("/notes")
def create_note(note: NoteCreate):
    note_id = len(notes_db) + 1
    notes_db[note_id] = note.dict()
    return {"id": note_id, "message": "Note created successfully"}

@router.get("/notes")
def get_all_notes():
    return {"notes": notes_db}

@router.get("/notes/{note_id}")
def get_note(note_id: int):
    note = notes_db.get(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note

@router.put("/notes/{note_id}")
def update_note(note_id: int, updated_note: NoteUpdate):
    if note_id not in notes_db:
        raise HTTPException(status_code=404, detail="Note not found")
    notes_db[note_id].update(updated_note.dict(exclude_unset=True))
    return {"message": "Note updated successfully"}

@router.delete("/notes/{note_id}")
def delete_note(note_id: int):
    if note_id not in notes_db:
        raise HTTPException(status_code=404, detail="Note not found")
    del notes_db[note_id]
    return {"message": "Note deleted successfully"}
