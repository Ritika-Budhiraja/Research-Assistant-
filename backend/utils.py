import fitz  # PyMuPDF
import os


from langchain.text_splitter import RecursiveCharacterTextSplitter

# -------------------------------
# Extract Text from File (PDF/TXT)
# -------------------------------

def extract_text_from_file(uploaded_file, file_ext):
    """
    Extracts raw text from PDF or TXT file.
    """
    if file_ext.lower() == ".pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_ext.lower() == ".txt":
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Only PDF and TXT are allowed.")

def extract_text_from_pdf(file):
    """
    Extracts text from a PDF file using PyMuPDF.
    """
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_txt(file):
    """
    Extracts text from a plain TXT file.
    """
    return file.read().decode("utf-8")

# -------------------------------
# Text Chunking for Vector Indexing
# -------------------------------

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits text into overlapping chunks for vector embedding and retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)
