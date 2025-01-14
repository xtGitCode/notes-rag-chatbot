import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- ChromaDB Configuration ---
CHROMA_DB_DIR = os.path.join(os.path.expanduser("~"), "my_chroma_db")
os.makedirs(CHROMA_DB_DIR, exist_ok=True)  # Ensure directory exists

try:
    client = chromadb.Client(Settings(
        persist_directory=CHROMA_DB_DIR,
        is_persistent=True))
    logger.info(f"ChromaDB client initialized. Data directory: {CHROMA_DB_DIR}")
except Exception as e:
    logger.error(f"Error initializing ChromaDB client: {e}")
    raise  # Re-raise the exception to stop execution

COLLECTION_NAME = "notes"
try:
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    logger.info(f"ChromaDB collection '{COLLECTION_NAME}' accessed/created.")
except Exception as e:
    logger.error(f"Error getting/creating ChromaDB collection: {e}")
    raise

# --- Embedding Model ---
try:
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    logger.info("SentenceTransformer model loaded.")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {e}")
    raise


def add_note_to_chroma(note_title: str, note_content: str, metadata: dict = None):
    """Adds a note to ChromaDB."""
    try:
        note_id = str(uuid.uuid4())
        embeddings = embedding_model.encode([note_content]).tolist()

        collection.add(
            documents=[note_content],  # The actual text content for search
            ids=[note_id],
            embeddings=embeddings,
            metadatas=[{"title": note_title, **(metadata or {})}] # Add metadata as a dictionary
        )

        logger.info(f"Note with ID {note_id} added successfully.")
        return {"id": note_id, "title": note_title, "content": note_content} # Return a dictionary
    except Exception as e:
        logger.error(f"Error adding note to ChromaDB: {e}")
        return None  # Return None on failure

def query_chroma(query: str, n_results: int = 3):
    """Queries ChromaDB.

    Args:
        query: Query string.
        n_results: Number of results to retrieve.

    Returns:
        Query results or None on error.
    """
    try:
        query_embedding = embedding_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=n_results)
        return results
    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return None

def delete_note_from_chroma(note_id: int):
    """Deletes a note from ChromaDB.

    Args:
        note_id: ID of the note to delete.
    Returns:
        True on success, False on failure
    """
    try:
        collection.delete(ids=[str(note_id)])
        logger.info(f"Note with ID {note_id} deleted successfully.")
        return True
    except Exception as e:
        logger.error(f"Error deleting note from ChromaDB: {e}")
        return False