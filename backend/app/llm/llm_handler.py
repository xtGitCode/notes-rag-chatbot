import os
import torch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.prompts import PromptTemplate
from .prompt_templates import get_rag_prompt

class NotesRAGBot:
    def __init__(self, chroma_db_path="./chroma_db"): # add a parameter for chroma db path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        token = os.environ.get("HF_TOKEN") # get token from env variable
        if token is None:
            raise ValueError("HF_TOKEN environment variable not set")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=token,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True
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
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        self.db = None
        self.chain = None
        self.chroma_db_path = chroma_db_path # store chroma db path

    def add_notes(self, notes_text):
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(notes_text)
        self.db = Chroma.from_texts(chunks, self.embeddings, persist_directory=self.chroma_db_path)
        rag_prompt = get_rag_prompt() 
        self.chain = ConversationalRetrievalChain.from_llm(
            self.llm, retriever=self.db.as_retriever(search_kwargs={"k": 3}), memory=self.memory, return_source_documents=True, prompt=rag_prompt
        )
        return f"Successfully processed {len(chunks)} chunks of notes"

    def query(self, question):
        if not self.chain:
            return "Please add some notes first!"

        chat_history = self.memory.load_memory_variables({})["chat_history"]
        total_tokens = sum(len(self.tokenizer.encode(msg.content)) for msg in chat_history) 

        max_allowed_tokens = 1900
        while total_tokens > max_allowed_tokens and chat_history: 
            chat_history.pop(0)
            total_tokens = sum(len(self.tokenizer.encode(msg.content)) for msg in chat_history)

        self.memory.chat_memory.messages = chat_history
        response = self.chain({"question": question})
        return response["answer"]