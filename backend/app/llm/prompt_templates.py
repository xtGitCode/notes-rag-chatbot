from langchain.prompts import PromptTemplate

def get_rag_prompt():
    """Returns the prompt template for the RAG chatbot."""
    prompt_template = """
    You are a helpful assistant. Answer the question concisely based on the given notes. If you don't know the answer, just say that you don't know, don't try to make up an answer.:
    {context}

    Question: {question}
    Answer:
    """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])