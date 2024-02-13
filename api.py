from fastapi import FastAPI,Form,File,UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
from helper_functions import get_qa_chain,get_gemini_response,get_url_doc_qa,extract_transcript_details,get_gemini_response_health
from settings import invoice_prompt,youtube_transcribe_prompt
import google.generativeai as genai

app = FastAPI(title="Generative AI APIs",
              summary="This API contains routes of different Gen AI usecases")

class ResponseText(BaseModel):
    response: str


@app.get("/")
def home():
    return {"welcome":"Generative AI usecases API"}

@app.post("/invoice_extractor",description="This route extracts information from invoices based on provided images and prompts.")
def gemini(image_file: UploadFile = File(...), prompt: str = Form(...)):
    image = image_file.file.read()
    image_parts = [{
            "mime_type": "image/jpeg",
            "data": image
        }]
    output = get_gemini_response(invoice_prompt, image_parts, prompt)
    return ResponseText(response=output)

@app.post("/qa_from_faqs",description="The endpoint uses the retrieved question-answer to generate a response to the user's prompt")
def question_answer(prompt: str = Form(...)):
    try:
        chain = get_qa_chain()
        out = chain.invoke(prompt)
        return ResponseText(response=out["result"])
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")

@app.post("/qa_url_doc",description="In this route just add the doc or  url(of any news article,blogs etc) and then ask the question in the prompt ")
def qa_url_doc(url: list = Form(None), documents: List[UploadFile] = File(None), prompt: str = Form(...)):
    try:
        if url:
            chain = get_url_doc_qa(url,documents)
        elif documents:
            contents = [i.file.read().decode("utf-8") for i  in documents ]
            print(contents)
            # contents = documents.file.read().decode("utf-8")
            chain = get_url_doc_qa(url,contents)
        else:
            raise Exception("Please provide either a URL or upload a document file.")
        out = chain.invoke(prompt)
        return ResponseText(response=out["result"])
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")

@app.post("/youtube_video_transcribe_summarizer",description="The endpoint uses Youtube URL to generate a summary of a video")
def youtube_video_transcribe_summarizer_gemini(url: str = Form(...)):
    try:
        model = genai.GenerativeModel("gemini-pro")
        transcript_text = extract_transcript_details(url)
        response=model.generate_content(youtube_transcribe_prompt+transcript_text)
        return ResponseText(response=response.text)
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")
    
@app.post("/nutritionist_expert",description="This route need image,height(cm),weight(kg) then it extracts edible objects from image and return breif about calories to burn")
def health_app_gemini(image_file: UploadFile = File(...), height: str = Form(165),weight:str = Form(70)):
    image = image_file.file.read()
    image_parts = [{
            "mime_type": "image/jpeg",
            "data": image
        }]
    health_prompt=f"""
               You are an expert in nutritionist where you need to see the food items from the image
               and calculate the total calories,and if the person height {height} cm and weight is {weight} kg then it will loose or gain the weight and how much step count should be done to burn this calories.
               Also provide the details of every food items with calories intake is below format

               1. Item 1 - no of calories
               2. Item 2 - no of calories
               ----
               ----

            """
    output = get_gemini_response_health(image_parts, health_prompt)
    json_compatible_data = jsonable_encoder(output)
    return JSONResponse(content=json_compatible_data)
    
@app.post("/blog_generator",description="This route will generate the blog based on the desired topic.")
def blogs(topic: str = Form("Generative AI")):
    try:
        model = genai.GenerativeModel("gemini-pro")
        blog_prompt=f"""
               You are expert in blog writting. 
               write a blog on the topic {topic}. 
               Use a friendly and informative tone, and include examples and tips to encourage readers to get started with topic provided.
            """
        response=model.generate_content(blog_prompt)
        return ResponseText(response=response.text)
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")