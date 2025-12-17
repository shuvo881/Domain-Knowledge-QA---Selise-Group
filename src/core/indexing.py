import yaml
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from .model_emb import Loader
from .load_docs import DocumentProcessor

class VectorStoreManager:
    def __init__(self, config_path="./src/configs/config.yaml"):
        self.config = self._load_config(config_path)
        manager_config = self.config["vector_store_manager"]
        self.collection_name = manager_config["collection_name"]
        self.persist_directory = manager_config["persist_directory"]
        self.qdrant_client = QdrantClient(url="http://localhost:6333",)

    def _load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def _initialize_vector_store(self):
        try:
            loader = Loader()
            self.embeddings = loader.load_model_emb()
            if not self.embeddings:
                raise RuntimeError("Failed to initialize the embedding model.")

            if not self.qdrant_client.collection_exists(self.collection_name):
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.config["vector_store_manager"]["vector_size"], distance=Distance.COSINE)
                )
            vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )

            # vector_store = Chroma(
            #     collection_name=self.collection_name,
            #     embedding_function=self.embeddings,
            #     persist_directory=self.persist_directory,
            # )
            print(f"Vector store '{self.collection_name}' initialized successfully.")
            return vector_store
        except Exception as e:
            raise RuntimeError(f"Error initializing vector store: {e}")

    def index_documents(self):
        vector_store = self._initialize_vector_store()
        if not vector_store:
            raise RuntimeError("Vector store is not initialized. Call `initialize_vector_store` first.")

        try:
            processor = DocumentProcessor()
            splits = processor.process()
            if not splits:
                raise RuntimeError("Failed to split documents into chunks.")
            
            ids = vector_store.add_documents(documents=splits)
            if not ids:
                raise RuntimeError("Failed to add document chunks to the vector store.")
            
            return len(ids)
        except Exception as e:
            raise RuntimeError(f"Error during document indexing: {e}")
