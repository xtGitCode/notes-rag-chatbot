import os
import logging
import uuid
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "notes-index"

# Initialze pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
namespace = "notes-database"

# Check if the index exists; if not, create it
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(name=INDEX_NAME, dimension=768, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

index = pc.Index(INDEX_NAME)
print(index)

logger.info(f"Pinecone index '{INDEX_NAME}' initialized.")

# Embedding model
try:
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    logger.info("SentenceTransformer model loaded.")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {e}")
    raise

# Database operations
def add_note_to_pinecone(note_title: str, note_content: str, metadata: dict = None):
    """Adds a note to Pinecone."""
    try:
        note_id = str(uuid.uuid4())
        embedding = embedding_model.encode(note_content).tolist()
        
        # Include content in the metadata
        metadata_with_content = {
            "title": note_title,
            "content": note_content,
            **(metadata or {})
        }
        
        index.upsert(
            vectors=[(note_id, embedding, metadata_with_content)],
            namespace=namespace
        )
      
        logger.info(f"Note with ID {note_id} added successfully.")
        return {"id": note_id, "title": note_title, "content": note_content}
    except Exception as e:
        logger.error(f"Error adding note to Pinecone: {e}")
        return None

def delete_note_from_pinecone(note_id: str):
    """Deletes a note from Pinecone."""
    try:
        index.delete(ids=[note_id], namespace=namespace)
        logger.info(f"Note with ID {note_id} deleted successfully.")
        return True
    except Exception as e:
        logger.error(f"Error deleting note from Pinecone: {e}")
        return False

def query_pinecone(query_text: str, top_k=3):
    try:
        if not query_text.strip():
            logger.error("Query text is empty or invalid.")
            return []

        query_embedding = embedding_model.encode(query_text).tolist()

        # query pinecone index
        results = index.query(
            namespace=namespace,
            vector=query_embedding,
            top_k=top_k,
            include_values=True,
            include_metadata=True
        )

        if results.get("matches"):
            result_context = [
                {
                    "title": match["metadata"].get("title", "No Title"),
                    "content": match["metadata"].get("content", ""),
                    "score": match["score"]
                }
                for match in results["matches"]
            ]
        else:
            logger.warning("No matches found.")
            result_context = []

        return result_context

    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return []
