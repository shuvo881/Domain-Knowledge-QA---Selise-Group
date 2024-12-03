from ..pydentic_models.rag_model import State
from .indexing import VectorStoreManager


class DocumentRetrievalService:


    def __init__(self):

        try:
            self.vector_store_manager = VectorStoreManager()
            self.vector_store = self.vector_store_manager._initialize_vector_store()
            if not self.vector_store:
                raise RuntimeError("Failed to initialize the vector store.")
        except Exception as e:
            raise RuntimeError(f"Error initializing DocumentRetrievalService: {e}")

    def retrieve_context(self, state: State):
        try:
            retrieved_docs = self.vector_store.similarity_search(state["question"])
            if not retrieved_docs:
                raise ValueError("No documents were retrieved.")
            return {"context": retrieved_docs}
        except KeyError as e:
            raise ValueError(f"Missing required key in state: {e}")
        except Exception as e:
            raise RuntimeError(f"Error during document retrieval: {e}")
