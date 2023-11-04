from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from enum import Enum
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

test_txt = "Image of a cat"
test_img = "https://images.unsplash.com/photo-1608848461950-0fe51dfc41cb?auto=format&fit=crop&q=80&w=2487&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

test_question = "How are you today?"

class Story(BaseModel):
    page_text: str
    image: str


class StoryGenerateRequestBody(BaseModel):
    prompt: str


class StoryGenerateResponse(BaseModel):
    story: List[Story]
    first_question: str


@app.post("/story", response_model=StoryGenerateResponse)
async def generate(request_body: StoryGenerateRequestBody):
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
        "first_question": test_question
    }


class QuizResponseRequestBody(BaseModel):
    answer: str

class QuizResponseModel(BaseModel):
    image: str
    llm_response: str
    is_correct: bool
    next_question: str


@app.post("/quiz-response", response_model=QuizResponseModel)
async def quiz_response(request_body: QuizResponseRequestBody):
    return {
        "image": test_img,
        "llm_response": "Correct response",
        "is_correct": True,
        "next_question": ""
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
