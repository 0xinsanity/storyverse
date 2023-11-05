import os
import openai
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from enum import Enum
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

openai.api_key = os.environ['OPENAI_API_KEY']

chat_llm_model = "gpt-3.5-turbo-0301"
chat = ChatOpenAI(temperature=0.0, model=chat_llm_model)

image_llm = OpenAI(temperature=0.9)

def create_story(outline, age):
    template_string = """Generate a story for a {age} year old girl \
        which consists of two paragraphs with each paragraph four sentences long. \
        The story should be about the following: ```{outline}```
        In the story, there should be a dramatic situation and a happy ending. \
        Response MUST be in json format with each key and value being paragraph number \
        and text of each paragraph being its value.
    """

    prompt_template = ChatPromptTemplate.from_template(template_string)

    #prompt_template.messages[0].prompt
    story_prompt = prompt_template.format_messages(
                    age=age,
                    outline=outline)
    story_response = chat(story_prompt)

    story_response.content = json.loads(story_response.content)
    story = story_response.content
    story.update({'story': "\n".join(story.values())})

    return story

def _generate_image_description(story, age, scene):
    image_desc = """Given the following story context: {story}. \
        Generate an image description that MUST be relevant for a {age} year old and \
        for the following scene: {scene} and response MUST be string.
    """
    image_template = ChatPromptTemplate.from_template(image_desc)

    image_prompt = image_template.format_messages(
                    age=age,
                    scene=scene,
                    story=story)
    image_scene_desc = chat(image_prompt)
    return image_scene_desc.content

def generate_image_descriptions(story, age):
    image_descriptions = {}

    for i, scene in story.items():
        if i != 'story':
            image_desc = _generate_image_description(story, age, scene)
            image_descriptions[i] = image_desc

    return image_descriptions

def _generate_image(image_description, age):
    prompt = PromptTemplate(
        input_variables=["image_description"],
        template="""Generate the image so that it is in the style of a story book fit for 4 year old children. \
            The following is a description of a scene: {image_description}.
        """
    )
    chain = LLMChain(llm=image_llm, prompt=prompt)
    image_url = DallEAPIWrapper().run(chain.run(
        image_description
    ))
    
    return image_url

def generate_images(image_descriptions, age):
    images = {}
    for i, image_desc in image_descriptions.items():
        image_url = _generate_image(image_desc, age)
        images[i] = image_url
    
    return images

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://storyverse-l2tovtpv2-nhano0228.vercel.app"],  # Allows all origins
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
    age: str


class StoryGenerateResponse(BaseModel):
    story: List[Story]
    first_question: str


@app.post("/story", response_model=StoryGenerateResponse)
async def generate(request_body: StoryGenerateRequestBody):
    story = create_story(request_body.prompt, request_body.age)
    descriptions = generate_image_descriptions(story, request_body.age)
    images = generate_images(descriptions, request_body.age)

    return_val = {"story": [], "first_question": ""}

    for i in range(len(descriptions)):
        r = {
            "page_text": story["paragraph"+str(i+1)],
            "image": images["paragraph"+str(i+1)]
        }
        return_val["story"].append(r)
    
    return return_val


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
