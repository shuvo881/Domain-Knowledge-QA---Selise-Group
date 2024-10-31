
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.pipeline import generate_answer
import logging

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/answer")
async def answer(request: QuestionRequest):
    logging.info(f"Received question: {request.question}")
    
    print("Received question:", request.question)

    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        answer = generate_answer(request.question)
        return {"answer": answer}
    except Exception as e:
        logging.error(f"Error in processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
