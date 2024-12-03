from src.pydentic_models.rag_model import State
from .model_emb import Loader


class QuestionAnsweringService:
    
    def __init__(self):
        try:
            self.loader = Loader()
            self.llm = self.loader.load_model()
            if not self.llm:
                raise RuntimeError("Failed to load the language model.")
        except Exception as e:
            raise RuntimeError(f"Error initializing QuestionAnsweringService: {e}")

    def generate_answer(self, state: State):
        try:
            prompt_template = """
                You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
                Question: {question}
                Context: {context}
                Answer:
            """

            docs_content = "\n\n".join(doc.page_content for doc in state["context"])

            formatted_prompt = prompt_template.format(
                question=state['question'],
                context=docs_content
            )

            response = self.llm.invoke(formatted_prompt)
            if not response:
                raise RuntimeError("Failed to generate an answer.")
            
            return {"answer": response.content}

        except KeyError as e:
            raise ValueError(f"Missing required key in state: {e}")
        except Exception as e:
            raise RuntimeError(f"Error generating answer: {e}")


