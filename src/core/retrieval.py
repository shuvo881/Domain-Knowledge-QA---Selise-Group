from ..pydentic_models.rag_model import State
from .indexing import VectorStoreManager


class DocumentRetrievalService:


    def __init__(self):

        try:
            self.vector_store_manager = VectorStoreManager()
            self.config = self.vector_store_manager.config
            retrieval_config = self.config["retrieval"]
            self.num_chunks = retrieval_config["num_chunks"]
            self.min_relevance_score = retrieval_config["min_relevance_score"]
            self.vector_store = self.vector_store_manager._initialize_vector_store()
            if not self.vector_store:
                raise RuntimeError("Failed to initialize the vector store.")
        except Exception as e:
            raise RuntimeError(f"Error initializing DocumentRetrievalService: {e}")

    def retrieve_context(self, state: State):
        try:
            # Retrieve documents with similarity scores
            retrieved_docs = self.vector_store.similarity_search_with_score(
                state["question"], k=self.num_chunks
            )

            if not retrieved_docs:
                raise ValueError("No documents were retrieved.")

            print(f"Number of documents retrieved: {len(retrieved_docs)}\n")

            # Optional: store scores in metadata if needed
            context_list = []
            for i, (doc, score) in enumerate(retrieved_docs, start=1):
                doc.metadata["similarity_score"] = score
                if score >= self.min_relevance_score:
                    context_list.append(doc)

                # print(f"Document {i}")
                # print(f"Score: {score:.4f}")  # similarity score
                # print(f"Source: {doc.metadata.get('source')}")
                # print(f"Content snippet: {doc.page_content[:200]}...\n")

            print(f"Number of documents after score filtering: {len(context_list)}\n")

            return {"context": context_list}

        except KeyError as e:
            raise ValueError(f"Missing required key in state: {e}")
        except Exception as e:
            raise RuntimeError(f"Error during document retrieval: {e}")

