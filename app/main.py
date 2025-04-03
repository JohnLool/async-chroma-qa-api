from fastapi import FastAPI
from app.process_question import process_question
from app.schemas import QuestionRequest, QuestionResponse

app = FastAPI()


@app.post("/question", response_model=QuestionResponse)
async def question_endpoint(q: QuestionRequest):
    return await process_question(q)