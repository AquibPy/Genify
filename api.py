from fastapi import FastAPI,Form,File,UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse,RedirectResponse
from typing import List
from pydantic import BaseModel
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from settings import invoice_prompt,youtube_transcribe_prompt,text2sql_prompt,EMPLOYEE_DB
from mongo import MongoDB
from helper_functions import get_qa_chain,get_gemini_response,get_url_doc_qa,extract_transcript_details,\
    get_gemini_response_health,get_gemini_pdf,read_sql_query,remove_substrings,questions_generator

app = FastAPI(title="Genify By Mohd Aquib",
              summary="This API contains routes of different Gen AI usecases")

app.allow_dangerous_deserialization = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResponseText(BaseModel):
    response: str


@app.get("/", response_class=RedirectResponse)
def home():
    return "/docs"

@app.post("/invoice_extractor",description="This route extracts information from invoices based on provided images and prompts.")
async def gemini(image_file: UploadFile = File(...), prompt: str = Form(...)):
    image = image_file.file.read()
    image_parts = [{
            "mime_type": "image/jpeg",
            "data": image
        }]
    output = get_gemini_response(invoice_prompt, image_parts, prompt)
    db = MongoDB()
    payload = {
            "endpoint" : "/invoice_extractor",
            "prompt" : prompt,
            "output" : output
        }
    mongo_data = {"Document": payload}
    result = db.insert_data(mongo_data)
    print(result)
    return ResponseText(response=output)

@app.post("/qa_from_faqs",description="The endpoint uses the retrieved question-answer to generate a response to the user's prompt")
async def question_answer(prompt: str = Form(...)):
    try:
        chain = get_qa_chain()
        out = chain.invoke(prompt)
        db = MongoDB()
        payload = {
                "endpoint" : "/qa_from_faqs",
                "prompt" : prompt,
                "output" : out["result"]
            }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return ResponseText(response=out["result"])
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")

@app.post("/qa_url_doc",description="In this route just add the doc or  url(of any news article,blogs etc) and then ask the question in the prompt ")
async def qa_url_doc(url: list = Form(None), documents: List[UploadFile] = File(None), prompt: str = Form(...)):
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
        db = MongoDB()
        payload = {
                "endpoint" : "/qa_url_doc",
                "prompt" : prompt,
                "url": url,
                "documents":"If URl is null then they might have upload .txt file",
                "output" : out["result"]
            }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return ResponseText(response=out["result"])
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")

@app.post("/youtube_video_transcribe_summarizer",description="The endpoint uses Youtube URL to generate a summary of a video")
async def youtube_video_transcribe_summarizer_gemini(url: str = Form(...)):
    try:
        model = genai.GenerativeModel("gemini-pro")
        transcript_text = extract_transcript_details(url)
        response=model.generate_content(youtube_transcribe_prompt+transcript_text)
        db = MongoDB()
        payload = {
                "endpoint" : "/youtube_video_transcribe_summarizer",
                "url" : url,
                "output" : response.text
            }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return ResponseText(response=response.text)
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")
    
@app.post("/nutritionist_expert",description="This route need image,height(cm),weight(kg) then it extracts edible objects from image and return breif about calories to burn")
async def health_app_gemini(image_file: UploadFile = File(...), height: str = Form(165),weight:str = Form(70)):
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
    db = MongoDB()
    payload = {
            "endpoint" : "/nutritionist_expert",
            "height (in cms)" : height,
            "weight (in kgs)" : weight,
            "output" : json_compatible_data
        }
    mongo_data = {"Document": payload}
    result = db.insert_data(mongo_data)
    print(result)
    return ResponseText(response=json_compatible_data)
    
@app.post("/blog_generator",description="This route will generate the blog based on the desired topic.")
async def blogs(topic: str = Form("Generative AI")):
    try:
        model = genai.GenerativeModel("gemini-pro")
        blog_prompt=f"""
               You are expert in blog writting. 
               write a blog on the topic {topic}. 
               Use a friendly and informative tone, and include examples and tips to encourage readers to get started with topic provided.
            """
        response=model.generate_content(blog_prompt)
        db = MongoDB()
        payload = {
            "endpoint" : "/blog_generator",
            "topic" : topic,
            "output" : response.text
        }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return ResponseText(response=response.text)
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")

@app.post("/talk2PDF",description="The endpoint uses the pdf and give the answer based on the prompt provided")
async def talk_pdf(pdf: UploadFile = File(...),prompt: str = Form(...)):
    try:
        # contents = [i.file.read().decode("utf-8") for i  in pdf ]
        chain = get_gemini_pdf(pdf.file)
        out = chain.invoke(prompt)
        db = MongoDB()
        payload = {
            "endpoint" : "/talk2PDF",
            "prompt" : prompt,
            "output" : out["result"]
        }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return ResponseText(response=out["result"])
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")

@app.post("/Text2SQL",description="""This route will generate the SQL query and results from employees table based on the prompt given.
          \nColumns present in the table are Employee_ID, Name, Department, Title, Email, City, Salary, Work_Experience""")
async def sql_query(prompt: str = Form("Tell me the employees living in city Noida")):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response=model.generate_content([text2sql_prompt,prompt])
        output_query = remove_substrings(response.text)
        print(output_query)
        output = read_sql_query(remove_substrings(output_query),EMPLOYEE_DB)
        db = MongoDB()
        payload = {
            "endpoint" : "/Text2SQL",
            "prompt" : prompt,
            "SQL Query" : output_query,
            "output" : output
        }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return {"response" : {"SQL Query":output_query,"Data": output}}
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")

@app.post("/questions_generator",description="""The endpoint uses the pdf and generate the questions.
          \nThis will be helpful for the students or teachers preparing for their exams or test. """)
async def pdf_questions_generator(pdf: UploadFile = File(...)):
    try:
        out = questions_generator(pdf.file)
        db = MongoDB()
        payload = {
            "endpoint" : "/questions_generator",
            "output" : out
        }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return ResponseText(response=remove_substrings(out))
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")