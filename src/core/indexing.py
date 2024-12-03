from langchain_chroma import Chroma
from .model_emb import Loader
from .load_docs import load_docs, split_docs


class VectorStoreManager:
    
    def __init__(self, collection_name='example_collection', persist_directory="./src/data/processed"):

        self.collection_name = collection_name
        self.persist_directory = persist_directory

    def _initialize_vector_store(self):

        try:
            loader = Loader()
            self.embeddings = loader.load_model_emb()
            if not self.embeddings:
                raise RuntimeError("Failed to initialize the embedding model.")

            vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            print(f"Vector store '{self.collection_name}' initialized successfully.")
            return vector_store
        except Exception as e:
            raise RuntimeError(f"Error initializing vector store: {e}")

    def index_documents(self):
        vector_store = self._initialize_vector_store()
        if not vector_store:
            raise RuntimeError("Vector store is not initialized. Call `initialize_vector_store` first.")

        try:
            # Load and split documents
            docs = load_docs()
            if not docs:
                raise RuntimeError("No documents were loaded. Please check the data source.")

            splits = split_docs(docs)
            if not splits:
                raise RuntimeError("Failed to split documents into chunks.")


            ids = vector_store.add_documents(documents=splits)
            if not ids:
                raise RuntimeError("Failed to add document chunks to the vector store.")
            
            return len(ids)
        except Exception as e:
            raise RuntimeError(f"Error during document indexing: {e}")


def main():
    """
    Main function to demonstrate the usage of VectorStoreManager.
    """
    try:
        # Initialize and use the VectorStoreManager
        manager = VectorStoreManager(collection_name="example_collection")
        chunks_indexed = manager.index_documents()
        print(f"Indexed {chunks_indexed} document chunks.")

        # Example of using the returned vector store directly

    except Exception as e:
        print(f"An error occurred: {e}")


# if __name__ == "__main__":
#     main()
