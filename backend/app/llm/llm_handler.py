import os
import torch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .prompt_templates import get_rag_prompt
from backend.app.database import CHROMA_DB_DIR, client, COLLECTION_NAME
from dotenv import load_dotenv

load_dotenv()

class NotesRAGBot:
    def __init__(self, chroma_db_path=CHROMA_DB_DIR):
        # Load embeddings model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Load LLM model
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError("HF_TOKEN environment variable not set")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=token,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_length=2048,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Set up conversation memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

        # Load ChromaDB collection
        try:
            self.db = Chroma(
                client=client, 
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings)
            print(f"Successfully loaded collection '{COLLECTION_NAME}' from ChromaDB.")
        except Exception as e:
            print(f"Error accessing ChromaDB collection: {e}")
            raise

        # Initialize chain variables
        self.chain = None
        self.chroma_db_path = chroma_db_path

    def query(self, question):
        """Processes the user's query and retrieves a response."""
        if not self.db:
            return "ChromaDB collection is not available!"
        
        # Initialize the chain if not already done
        if not self.chain:
            rag_prompt = get_rag_prompt()
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.db.as_retriever(search_kwargs={"k": 3}),
                memory=self.memory,
                return_source_documents=True,
            )
            print("ConversationalRetrievalChain initialized.")
        
        # Manage memory to stay within token limits
        chat_history = self.memory.chat_memory.messages
        max_allowed_tokens = 1900  # Tokens limit
        total_tokens = sum(len(self.tokenizer.encode(msg.content)) for msg in chat_history)

        while total_tokens > max_allowed_tokens and chat_history:
            chat_history.pop(0)
            total_tokens = sum(len(self.tokenizer.encode(msg.content)) for msg in chat_history)

        # Update memory with trimmed chat history
        self.memory.chat_memory.messages = chat_history

         # Add prompt manually to the input
        try:
            # Perform a similarity search to fetch the context
            relevant_context = self.db.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in relevant_context])

            # Use the RAG prompt
            formatted_prompt = rag_prompt.format(context=context, question=question)

            # Run the chain with the formatted prompt
            response = self.chain({"question": formatted_prompt})
            return response["answer"]
        except Exception as e:
            print(f"Error during query processing: {e}")
            return "An error occurred while processing your request."
