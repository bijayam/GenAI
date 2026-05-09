from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import os

pdf_path = "./data/AI_in_modern_society.pdf"

reader = PdfReader(pdf_path)

raw_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        raw_text += text + "\n"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.create_documents([raw_text])

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large"
)

vector_store = Chroma(
    collection_name="rag_collection",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

if vector_store._collection.count() == 0:
    vector_store.add_documents(
        documents=chunks,
        ids=[f"id_{i}" for i in range(len(chunks))]
    )

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)