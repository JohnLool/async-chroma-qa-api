from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.process_question import process_question
from app.schemas import QuestionRequest, QuestionResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/question", response_model=QuestionResponse)
async def question_endpoint(q: QuestionRequest):
    return await process_question(q)