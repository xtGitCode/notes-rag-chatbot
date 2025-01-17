import os
from typing import List, Optional
from transformers import AutoTokenizer
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from backend.app.database import CHROMA_DB_DIR, client, COLLECTION_NAME
from .prompt_templates import get_rag_prompt
from dotenv import load_dotenv

load_dotenv()

class RAGBotError(Exception):
    """Custom exception for RAGBot specific errors."""
    pass

class NotesRAGBot:
    def __init__(self, chroma_db_path: str = CHROMA_DB_DIR):
        """
        Initialize the RAG bot with necessary components.
        
        Args:
            chroma_db_path (str): Path to ChromaDB directory
        """
        self._setup_environment()
        self._initialize_components()
        self.chroma_db_path = chroma_db_path

    def _setup_environment(self) -> None:
        """Set up environment variables and configurations."""
        self.hf_token = os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise RAGBotError("HF_TOKEN environment variable not set")
        
        # Changed to a free model
        self.model_name = "google/flan-t5-base"  # Alternative options: "facebook/opt-350m", "EleutherAI/gpt-neo-125m"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_tokens = 512  # Adjusted for the smaller model

    def _initialize_components(self) -> None:
        """Initialize embeddings, LLM, database, and other components."""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )

            # Initialize LLM with adjusted parameters for the free model
            self.llm = HuggingFaceHub(
                repo_id=self.model_name,
                huggingfacehub_api_token=self.hf_token,
                model_kwargs={
                    "temperature": 0.7,
                    "max_length": 512,
                    "top_p": 0.95,
                    "do_sample": True
                }
            )

            # Rest of the initialization code remains the same
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

            # Initialize ChromaDB
            self._setup_database()

            # Initialize chain
            self.chain = None

        except Exception as e:
            raise RAGBotError(f"Error initializing components: {e}")

    def _setup_database(self) -> None:
        """Set up and connect to ChromaDB."""
        try:
            self.db = Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings
            )
            print(f"Successfully loaded collection '{COLLECTION_NAME}' from ChromaDB.")
        except Exception as e:
            raise RAGBotError(f"Error accessing ChromaDB collection: {e}")

    def _count_tokens(self, messages: List[str]) -> int:
        """
        Count tokens in a list of messages using the model's tokenizer.
        
        Args:
            messages (List[str]): List of message strings
            
        Returns:
            int: Total token count
        """
        return sum(len(self.tokenizer.encode(msg)) for msg in messages)

    def _manage_memory(self) -> None:
        """Manage conversation memory to stay within token limits."""
        chat_history = self.memory.chat_memory.messages
        messages = [msg.content for msg in chat_history]
        total_tokens = self._count_tokens(messages)

        while total_tokens > self.max_tokens and chat_history:
            chat_history.pop(0)
            messages = [msg.content for msg in chat_history]
            total_tokens = self._count_tokens(messages)

        self.memory.chat_memory.messages = chat_history

    def _get_relevant_context(self, question: str, k: int = 3) -> tuple[str, list]:
        """
        Retrieve relevant context from the vector store.
        
        Args:
            question (str): User's question
            k (int): Number of documents to retrieve
            
        Returns:
            tuple[str, list]: Combined context string and list of raw results
        """
        try:
            combined_docs = []
            for doc in self.db.get()['documents']: # Access documents directly
                title = self.db.get()['metadatas'][self.db.get()['documents'].index(doc)].get('title', '')
                combined_text = f"{title}\n{doc}"  # Combine title and content
                combined_docs.append(combined_text)

            # Get documents with their scores
            results = self.db.similarity_search_with_score(
                question,
                k=k
            )
            
            # Filter results based on score
            filtered_docs = [
                doc[0].page_content 
                for doc in results 
                if doc[1] < 1.5  # Lower score means more similar
            ]
            
            # If no results pass the threshold, take the best match
            if not filtered_docs and results:
                return None, []
                
            return "\n".join(filtered_docs), results
                
        except Exception as e:
            raise RAGBotError(f"Error retrieving context: {e}")
    
    def query(self, question: str) -> dict:
        """
        Process user query and generate response.
        
        Args:
            question (str): User's question
                
        Returns:
            dict: Response including answer and debug info
        """
        if not self.db:
            raise RAGBotError("ChromaDB collection is not available!")

        try:
            # Initialize chain if not already done
            if not self.chain:
                # Get the prompt template
                rag_prompt = get_rag_prompt()
                
                self.chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.db.as_retriever(search_kwargs={"k": 3}),
                    memory=self.memory,
                    return_source_documents=True,
                    combine_docs_chain_kwargs={"prompt": rag_prompt}  # Use the prompt template properly
                )

            # Manage conversation memory
            self._manage_memory()

            # Get relevant context and debug info
            context, raw_results = self._get_relevant_context(question)

            if context:
                # Generate response
                response = self.chain.invoke({"question": question})
                answer = response["answer"]

                # Add debug information
                debug_info = {
                    "retrieved_context": context,
                    "raw_results": [
                        {
                            "content": doc[0].page_content,
                            "score": doc[1],
                            "metadata": doc[0].metadata
                        }
                        for doc in raw_results
                    ]
                }

                return {
                    "answer": str(answer),
                    "title": raw_results[:1][0][0].metadata['title'],
                    "content": raw_results[:1][0][0].page_content
                }
            
            else:
                return {"answer": "I couldn't find anything relevant in your notes."}

        except Exception as e:
            raise RAGBotError(f"Error processing query: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        try:
            if self.db:
                self.db.persist()
        except Exception as e:
            print(f"Error during cleanup: {e}")