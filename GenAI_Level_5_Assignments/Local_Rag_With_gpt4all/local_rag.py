
# Fully Local RAG with GPT4All and ChromaDB
# GPT4All → local LLM
# ChromaDB → local vector DB
# SentenceTransformers → local embeddings

# Workflow:
#           PDF Files
#               ↓
#       SentenceTransformer Embeddings
#               ↓
#           ChromaDB
#               ↓  <--------------------------User Query
#       Relevant Chunks Retrieved
#               ↓
#       GPT4All Local Llama Model
#               ↓
#       Answer Generated

from gpt4all import GPT4All
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import os

# -----------------------------
# STEP 1: Load local Embedding Model all-mpnet-base-v2 from HF
# -----------------------------
embedding_model = SentenceTransformer(
    #r"C:\BIJAY\AIModels\all-mpnet-base-v2"
    r"C:\BIJAY\AIModels\nomic-embed-text-v1.5"
)


# -----------------------------
# STEP 2: Create ChromaDB Client
# -----------------------------
DB_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = chroma_client.get_or_create_collection(
    name="rag_collection"
)

# -----------------------------
# STEP 3: Read all the PDF Documents in the "documents" folder,
# create embeddings, and add to ChromaDB.
# -----------------------------
DOCUMENTS_FOLDER = "./documents"

# Build DB only first time. 
# Otherwise, use existing DB. This way we don't have to 
# reprocess PDFs every time we run the script.
if collection.count() == 0:

    print("Creating ChromaDB...")

    doc_id = 0

    for filename in os.listdir(DOCUMENTS_FOLDER):

        if filename.endswith(".pdf"):

            pdf_path = os.path.join(DOCUMENTS_FOLDER, filename)

            reader = PdfReader(pdf_path)

            text = ""

            for page in reader.pages:
                text += page.extract_text()

            # Split into chunks
            chunks = text.split(". ")

            for chunk in chunks:

                if len(chunk.strip()) > 30:

                    # Create embedding for the chunk
                    embedding = embedding_model.encode(chunk).tolist()

                    # Add chunks to ChromaDB
                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        ids=[str(doc_id)]
                    )

                    doc_id += 1

    print("Documents added to ChromaDB")

else:
    print("Using existing ChromaDB")

# -----------------------------
# STEP 4: Load local GPT4All Model: Llama-3.2-3B-Instruct
# -----------------------------
model = GPT4All(
    model_name=r"C:\BIJAY\AIModels\GPT4All\Llama-3.2-3B-Instruct-Q4_0.gguf",
    device="cpu"
)

# -----------------------------
# STEP 5: Ask Questions
# -----------------------------
while True:

    query = input("\nAsk a question: or type 'exit' to quit: ")

    if query.lower() == "exit":
        break

    # Create query embedding
    query_embedding = embedding_model.encode(query).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    retrieved_docs = results['documents'][0]

    context = "\n".join(retrieved_docs)

    prompt = f"""
    You are a helpful AI assistant.

    Answer the question ONLY using the provided context.

    If the answer is not found in the context, say:
    "I could not find the answer in the provided documents."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # Generate response
    response = model.generate(
        prompt,
        max_tokens=300
    )

    print("\nAnswer:")
    print(response)