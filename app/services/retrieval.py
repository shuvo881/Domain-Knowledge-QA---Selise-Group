
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_path="app/models/faiss_index"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def embedding_function(text):
        return model.encode(text, convert_to_tensor=True)  # Ensure embedding returns a tensor
    
    vector_store = FAISS.load_local(index_path, embeddings=embedding_function, allow_dangerous_deserialization=True)
    return vector_store.as_retriever()

retriever = load_faiss_index()

def retrieve_documents(query: str):
    retrieved_docs = retriever.invoke(query)
    
    print(f"Type of retrieved_docs: {type(retrieved_docs)}")
    if isinstance(retrieved_docs, list) and len(retrieved_docs) > 0:
        print(f"Structure of first document: {retrieved_docs[0]}")
    else:
        print("No documents retrieved or unexpected format.")

    return retrieved_docs
