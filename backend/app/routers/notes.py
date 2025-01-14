from fastapi import APIRouter, HTTPException
from backend.app.models import NoteCreate, Note
from backend.app.database import add_note_to_chroma, client, collection, delete_note_from_chroma
from typing import List

router = APIRouter()
notes_db = {}
note_id_counter = 1

@router.post("/notes", response_model=Note, status_code=201)
def create_note(note: NoteCreate):
    global note_id_counter
    note_id = note_id_counter
    note_id_counter += 1

    notes_db[note_id] = note.dict()
    add_note_to_chroma(note_id, note.content)
    return Note(id=note_id, **note.dict())

@router.get("/notes", response_model=List[Note])
def get_all_notes():
    return [Note(id=id, **note) for id, note in notes_db.items()]

@router.get("/notes/{note_id}", response_model=Note)
def get_note(note_id: int):
    note = notes_db.get(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return Note(id=note_id, **note)

@router.delete("/notes/{note_id}", status_code=204)
def delete_note(note_id: int):
    if note_id not in notes_db:
        raise HTTPException(status_code=404, detail="Note not found")
    del notes_db[note_id]
    try:
        delete_note_from_chroma(note_id) #delete from chroma too
    except Exception as e:
        print(f"error deleting from chroma: {e}")
    return

@router.get("/chroma/count")
def get_chroma_count():
    return {"count": collection.count()}