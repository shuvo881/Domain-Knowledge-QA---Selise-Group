# scripts/setup_index.py

import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from app.utils.preprocess import clean_text, chunk_text

def prepare_documents(directory: str, text_column: str = "text"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    documents = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            

            if os.stat(file_path).st_size == 0:
                print(f"Skipping empty file: {filename}")
                continue
            
            try:

                df = pd.read_csv(file_path, delimiter=",", quotechar='"', engine="python", on_bad_lines="skip")
                

                if df.empty:
                    raise pd.errors.EmptyDataError
                
            except pd.errors.EmptyDataError:
                print(f"Empty or unstructured data in {filename}. Attempting line-by-line fallback.")
                

                with open(file_path, 'r', encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        cleaned_text = clean_text(line)
                        chunks = chunk_text(cleaned_text)
                        documents.extend(chunks)
                continue
            except pd.errors.ParserError as e:
                print(f"Error reading {filename}: {e}")
                continue


            if text_column in df.columns:
                column_to_use = text_column
            else:
                column_to_use = df.columns[0]
                print(f"Column '{text_column}' not found in {filename}. Using first available column: '{column_to_use}'")
            

            for text in df[column_to_use].dropna():
                cleaned_text = clean_text(str(text))
                chunks = chunk_text(cleaned_text)
                documents.extend(chunks)

    print(f"Number of documents processed: {len(documents)}")


    embeddings = [model.encode(doc) for doc in documents]
    print(f"Number of embeddings generated: {len(embeddings)}")


    text_embeddings = list(zip(documents, embeddings))
    if len(text_embeddings) == 0:
        print("No text embeddings were generated. Please check the input data.")
        return
    

    vector_store = FAISS.from_embeddings(text_embeddings, embedding=model)
    vector_store.save_local("app/models/faiss_index")
    print("FAISS index created and saved.")

prepare_documents("app/data/documents/")
