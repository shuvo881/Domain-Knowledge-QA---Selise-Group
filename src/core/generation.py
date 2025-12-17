from langchain_core.prompts import ChatPromptTemplate
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
            

            docs_content = "\n\n".join(doc.page_content for doc in state["context"])

            prompt = ChatPromptTemplate.from_messages([
                ("human",
                """You are a factual assistant for question-answering tasks. 
                Use only the information provided in the context to answer the question. 
                Do not invent, guess, or hallucinate any information. 
                If the answer is not in the context, respond with 'I don't know'. 
                Keep the answer concise, maximum three sentences.

                Question: {question}
                Context: {context}

                Answer:""")
            ])

            messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = self.llm.invoke(messages)
            if not response:
                raise RuntimeError("Failed to generate an answer.")
            
            return {"answer": response.content}

        except KeyError as e:
            raise ValueError(f"Missing required key in state: {e}")
        except Exception as e:
            raise RuntimeError(f"Error generating answer: {e}")


