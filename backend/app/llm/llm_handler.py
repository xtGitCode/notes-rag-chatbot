import os
from typing import List
from transformers import AutoTokenizer
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from backend.app.database import index, query_pinecone, namespace
from .prompt_templates import get_rag_prompt
from langchain.docstore.document import Document
from langchain.vectorstores import Pinecone as PineconeStore

class RAGBotError(Exception):
    """Custom exception for RAGBot specific errors."""
    pass

class NotesRAGBot:
    def __init__(self):
        """
        Initialize the RAG bot with necessary components.
        """
        self._setup_environment()
        self._initialize_components()

    def _setup_environment(self) -> None:
        """Set up environment variables and configurations."""
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise RAGBotError("HF_TOKEN environment variable not set")
        
        self.model_name = "google/flan-t5-base" # free model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.max_tokens = 512 

    def _initialize_components(self) -> None:
        """Initialize embeddings, LLM, database, and other components."""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )

            # Initialize LLM
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

            # Initialize pinecone db
            self.vectordb = PineconeStore(index, self.embeddings, text_key="content", namespace=namespace)

            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

            # Initialize chain
            self.chain = None

        except Exception as e:
            raise RAGBotError(f"Error initializing components: {e}")

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

    def _get_relevant_context(self, question: str, k: int = 3) -> List[Document]:
        """Retrieve relevant context from Pinecone and return as a list of Documents."""
        try:
            results = query_pinecone(question, k)

            # Filter results
            score_threshold = 0.5  # based on trial and error
            filtered_docs = [
                result  
                for result in results
                if isinstance(result, dict) and result.get('score', float('inf')) >= score_threshold
            ]

            # Did noot pass filter
            if not filtered_docs:
                return None
            # Pass filter
            else:
                documents = [
                    Document(page_content=doc.get('content', ''), metadata=doc)
                    for doc in filtered_docs
                ]
                return documents

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
        try:
            # Initialize chain if not already done
            if not self.chain:
                # Get prompt template from pprompt_templates.py
                rag_prompt = get_rag_prompt()

                self.chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vectordb.as_retriever(search_kwargs={'k': 10}),
                    memory=self.memory,
                    return_source_documents=True,
                    combine_docs_chain_kwargs={"prompt": rag_prompt},
                    chain_type="stuff",
                    verbose=True
                )

            # Manage conversation memory
            self._manage_memory()

            # Get relevant context to check if query pass the filter
            context = self._get_relevant_context(question)
            if context:   
                # if yes, invoke llm using self.chain             
                response = self.chain({"question": question})
                
                # access source_documents to extract response
                source_documents = response.get('source_documents', [])
                if source_documents:
                    first_document = source_documents[0]
                    title = first_document.metadata.get('title', 'No title available')
                    content = first_document.page_content
                else:
                    title = 'No title available'
                    content = 'No content available'

                return {
                    "answer": str(response["answer"]),
                    "title": title,
                    "content": content
                }
            # if no, no need to invoke llm
            else:
                return {"answer": "I couldn't find anything relevant in your notes.",
                        "title":"None",
                        "content":"None"}

        except Exception as e:
            raise RAGBotError(f"Error processing query: {e}")