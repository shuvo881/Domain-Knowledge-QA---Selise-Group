from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import time
from src.core.pipeline import QuestionAnsweringPipeline
from src.pydentic_models.rag_model import State 
app = FastAPI()

pipeline = QuestionAnsweringPipeline(State)

@app.get("/stream")
async def stream_endpoint(question: str):
    def event_generator():
        for chunk in pipeline.stream_responses(question):
            yield f"{chunk}"  # SSE format
    return StreamingResponse(event_generator(), media_type="text/event-stream")
