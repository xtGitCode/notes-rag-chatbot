import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings

client = chromadb.Client(Settings(
    persist_directory="./chroma_data",  
    chroma_db_impl="duckdb+parquet"
))

# Ensure the collection exists or create it
collection_name = "notes"
collection = client.get_or_create_collection(name=collection_name)

embedding_model = SentenceTransformer('all-mpnet-base-v2') # You can choose a different model

def add_note_to_chroma(note_id: int, note_content: str):
    """
    Add a note to ChromaDB with its embedding.
    
    :param note_id: Unique ID for the note
    :param note_content: Content of the note
    """
    try:
        embeddings = embedding_model.encode([note_content]).tolist()
        collection.add(
            documents=[note_content],
            ids=[str(note_id)],  # Chroma requires string IDs
            embeddings=embeddings,
            metadatas=[{"note_id": note_id}]
        )
        print(f"Note with ID {note_id} added successfully.")
    except Exception as e:
        print(f"Error adding note to ChromaDB: {e}")

def query_chroma(query: str, n_results: int = 3):
    """
    Query ChromaDB to retrieve the most relevant documents.
    
    :param query: Query string
    :param n_results: Number of results to retrieve (default: 3)
    :return: Query results
    """
    try:
        query_embedding = embedding_model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=n_results)
        return results
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return None

def delete_note_from_chroma(note_id: int):
    """
    Delete a note from ChromaDB using its ID.
    
    :param note_id: Unique ID for the note to delete
    """
    try:
        collection.delete(ids=[str(note_id)])
        print(f"Note with ID {note_id} deleted successfully.")
    except Exception as e:
        print(f"Error deleting note from ChromaDB: {e}")