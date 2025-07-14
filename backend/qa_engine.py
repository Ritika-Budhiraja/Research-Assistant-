import os
from dotenv import load_dotenv
import openai

from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

from utils import chunk_text

# ✅ Load API key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def build_vectorstore(text):
    """
    Splits the input text, generates embeddings, and builds a FAISS vectorstore.
    Returns the vectorstore object.
    """
    if not text:
        raise ValueError("Text input to build_vectorstore is empty.")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No text chunks generated for vectorstore.")

    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = OpenAIEmbeddings()

    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"Error building vectorstore: {str(e)}")


def answer_query(query, vectorstore, k=3):
    """
    Uses RetrievalQA to answer a question from top-k relevant chunks.
    Returns a dict with the answer and the reference content.
    """
    if not vectorstore:
        raise ValueError("Vectorstore is not initialized.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo"),
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain({"query": query})

    # Get justification from source documents
    sources = result.get("source_documents", [])
    source_chunks = [doc.page_content.strip() for doc in sources]
    justification = "\n---\n".join(source_chunks[:k]) if source_chunks else "No source chunks found."

    return {
        "answer": result["result"],
        "source": justification
    }
