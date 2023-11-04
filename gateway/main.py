from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from enum import Enum
import uvicorn

app = FastAPI()

test_txt = "Image of a cat"
test_img = "https://images.unsplash.com/photo-1608848461950-0fe51dfc41cb?auto=format&fit=crop&q=80&w=2487&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

test_question = "How are you today?"
test_a = "Good"
test_b = "Bad"
test_c = "Great"
test_d = "Awful"

class AnswerChoice(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class Quiz(BaseModel):
    question: str
    choice_a: str
    choice_b: str
    choice_c: str
    choice_d: str
    answer: AnswerChoice


class Story(BaseModel):
    page_text: str
    image: str


class GenerateRequestBody(BaseModel):
    prompt: str


class GenerateResponse(BaseModel):
    story: List[Story]
    quiz: List[Quiz]


@app.post("/story", response_model=GenerateResponse)
async def generate(request_body: GenerateRequestBody):
    # Access the prompt with request_body.prompt
    # Replace the following line with your logic to generate story and quiz
    return {
        "story": [
            {
                "page_text": test_txt,
                "image": test_img,
            },
            {
                "page_text": test_txt,
                "image": test_img,
            },
            {
                "page_text": test_txt,
                "image": test_img,
            },
            {
                "page_text": test_txt,
                "image": test_img,
            },
        ],
        "quiz": [
            {
                "question": test_question,
                "choice_a": test_a,
                "choice_b": test_b,
                "choice_c": test_c,
                "choice_d": test_d,
                "answer": AnswerChoice.A,
            },
            {
                "question": test_question,
                "choice_a": test_a,
                "choice_b": test_b,
                "choice_c": test_c,
                "choice_d": test_d,
                "answer": AnswerChoice.B,
            },
            {
                "question": test_question,
                "choice_a": test_a,
                "choice_b": test_b,
                "choice_c": test_c,
                "choice_d": test_d,
                "answer": AnswerChoice.C,
            },
        ]
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
