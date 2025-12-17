from src.core.pipeline import QuestionAnsweringPipeline
from src.pydentic_models.rag_model import State


if __name__ == "__main__":
    input_message = "Tell about Md. Golam Mostofa"

    # Initialize the pipeline with the State model
    pipeline = QuestionAnsweringPipeline(State)
    # Stream and print responses from the pipeline
    for response in pipeline.stream_responses(input_message):
        print(response, end="")

    print()  # For a clean newline after the streaming output