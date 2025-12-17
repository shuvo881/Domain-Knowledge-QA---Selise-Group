from langgraph.graph import START, StateGraph
from .retrieval import DocumentRetrievalService 
from .generation import QuestionAnsweringService
from ..pydentic_models.rag_model import State
from .critic import Critic


class QuestionAnsweringPipeline:

    def __init__(self, model: State):

        self.graph_builder = StateGraph(model)
        self._build_pipeline()
        self.graph = self.graph_builder.compile()

    def _build_pipeline(self):
        retrieve = DocumentRetrievalService().retrieve_context
        generate = QuestionAnsweringService().generate_answer
        self.graph_builder.add_sequence([retrieve, generate])
        self.graph_builder.add_edge(START, retrieve.__name__)

    def run(self, input_question: str) -> State:
        return self.graph.run({"question": input_question})

    def stream_responses(self, input_question: str):
        print("Streaming responses...")
        for message, metadata in self.graph.stream(
            {"question": input_question},
            stream_mode="messages",
        ):
            yield message.content



