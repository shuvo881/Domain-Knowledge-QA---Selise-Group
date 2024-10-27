from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from app.utils.preprocess import clean_text, chunk_text
import os

def prepare_documents(directory: str):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    documents = []

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as f:
                text = clean_text(f.read())
                chunks = chunk_text(text)
                documents.extend(chunks)
                
    embeddings = [model.encode(doc) for doc in documents]
    vector_store = FAISS.from_embeddings(embeddings)
    vector_store.save_local("app/models/faiss_index")
    print("FAISS index created and saved.")

# Run setup to create index
prepare_documents("app/data/documents/")
