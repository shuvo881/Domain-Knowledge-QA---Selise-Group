from src.core.indexing import VectorStoreManager


def main():
    """
    Main function to demonstrate the usage of VectorStoreManager.
    """
    try:
        # Initialize and use the VectorStoreManager
        manager = VectorStoreManager()
        chunks_indexed = manager.index_documents()
        print(f"Indexed {chunks_indexed} document chunks.")

        # Example of using the returned vector store directly

    except Exception as e:
        print(f"An error occurred: {e}")
         

if __name__ == "__main__":
    main()
