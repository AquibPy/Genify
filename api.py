import os
import uvicorn
import io
import requests
from PIL import Image
from redis import Redis
import json
from fastapi import FastAPI,Form,File,UploadFile, Request ,Response, HTTPException, status, Depends
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse,RedirectResponse,StreamingResponse
from typing import List,Optional
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from mongo import MongoDB
from helper_functions import get_qa_chain,get_gemini_response,get_url_doc_qa,extract_transcript_details,\
    get_gemini_response_health,get_gemini_pdf,read_sql_query,remove_substrings,questions_generator,groq_pdf,\
    summarize_audio,chatbot_send_message,extraxt_pdf_text,advance_rag_llama_index,parse_sql_response, extract_video_id
from langchain_groq import ChatGroq
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from auth import create_access_token
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import timedelta
from jose import jwt, JWTError
import settings
from models import UserCreate, ResponseText
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from uuid import uuid4
# from tech_news_agent.crew import run_crew
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_cohere.chat_models import ChatCohere
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import tempfile
import shutil
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]="genify"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"

redis = Redis(host=os.getenv("REDIS_HOST"), port=settings.REDIS_PORT, password=os.getenv("REDIS_PASSWORD"))

mongo_client = MongoDB(collection_name=os.getenv("MONGO_COLLECTION_USER"))
users_collection = mongo_client.collection
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="Genify By Mohd Aquib",
              summary="This API contains routes of different Gen AI usecases")

limiter = Limiter(key_func=get_remote_address)
app.state.limit = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request:Request,exc: RateLimitExceeded):
    return JSONResponse(
        status_code= status.HTTP_429_TOO_MANY_REQUESTS,
        content= {"response": "Limit exceeded, please try again after 2 minute !!!!!!"}
    )

templates = Jinja2Templates(directory="templates")

app.allow_dangerous_deserialization = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=RedirectResponse)
async def home():
    return RedirectResponse("/docs")

@app.post("/signup")
async def signup(user: UserCreate):
    # Check if user already exists
    existing_user = users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Insert new user to database with email_verified set to False
    verification_token = str(uuid4())
    new_user = {
        "email": user.email,
        "password": user.password,
        "email_verified": False,
        "verification_token": verification_token
    }
    users_collection.insert_one(new_user)

    # Send verification email
    message = Mail(
        from_email='maquib100@myamu.ac.in',
        to_emails=user.email,
        subject='Verify your email',
        html_content=f'Please verify your email using this token: {verification_token}'
    )

    try:
        sendgrid_api = os.getenv("SENDGRID_API_KEY")
        sg = SendGridAPIClient(sendgrid_api)
        response = sg.send(message)
        print(response.status_code)
    except Exception as e:
        print(e.message)

    return {"message": "User created successfully. Please check your email to verify your account."}

@app.post("/verify-email")
async def verify_email(token: str):
    # Find user with the provided verification token
    user = users_collection.find_one({"verification_token": token})
    if not user:
        raise HTTPException(status_code=400, detail="Invalid token")

    # Mark the user's email as verified
    users_collection.update_one({"_id": user["_id"]}, {"$set": {"email_verified": True}})

    return {"message": "Email verified successfully"}

# Signin route
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Check if user exists in database
    user = users_collection.find_one({"email": form_data.username})
    if not user or user["password"] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/chatbot",description="Provides a simple web interface to interact with the chatbot")
async def chat(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/blog_generator_ui",description="Provides a simple web interface to interact with the Blog Generator")
async def blog_ui(request: Request):
    return templates.TemplateResponse("blog_generator.html", {"request": request})

@app.get("/ats",description="Provides a simple web interface to interact with the Smart ATS")
async def ats(request: Request):
    return templates.TemplateResponse("ats.html", {"request": request})

@app.post("/invoice_extractor",description="This route extracts information from invoices based on provided images and prompts.")
async def gemini(image_file: UploadFile = File(...), prompt: str = Form(...)):
    image = image_file.file.read()
    image_parts = [{
            "mime_type": "image/jpeg",
            "data": image
        }]
    output = get_gemini_response(settings.invoice_prompt, image_parts, prompt)
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
        # Check if the response is cached in Redis
        cache_key = f"qa_from_faqs:{prompt}"
        cached_response = redis.get(cache_key)
        if cached_response:
            print("Retrieving response from Redis cache")
            out = {"result": cached_response.decode("utf-8")}
        else:
            print("Fetching response from the API")
            chain = get_qa_chain()
            out = chain.invoke(prompt)
            redis.set(cache_key, out["result"], ex=60)
            db = MongoDB()
            payload = {"endpoint": "/qa_from_faqs", "prompt": prompt, "output": out["result"]}
            mongo_data = {"Document": payload}
            result = db.insert_data(mongo_data)
            print(result)
        return ResponseText(response=out["result"])
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")

@app.post("/qa_url_doc", description="In this route just add the doc or url(of any news article,blogs etc) and then ask the question in the prompt ")
async def qa_url_doc(url: list = Form(None), documents: List[UploadFile] = File(None), prompt: str = Form(...)):
    try:
        if url:
            cache_key = f"qa_url_doc:{prompt}:{str(url)}"
            cached_response = redis.get(cache_key)
            if cached_response:
                print("Retrieving response from Redis cache")
                out = {"result": cached_response.decode("utf-8")}
                return ResponseText(response=out["result"])
            else:
                chain = get_url_doc_qa(url, documents)
                out = chain.invoke(prompt)
                redis.set(cache_key, out["result"], ex=60)
        else:
            if documents:
                contents = [i.file.read().decode("utf-8") for i in documents]
                print(contents)
                chain = get_url_doc_qa(url, contents)
            else:
                raise Exception("Please provide either a URL or upload a document file.")
            out = chain.invoke(prompt)

        db = MongoDB()
        payload = {
            "endpoint": "/qa_url_doc",
            "prompt": prompt,
            "url": url,
            "documents": "If URL is null then they might have upload .txt file",
            "output": out["result"]
        }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return ResponseText(response=out["result"])
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")

@app.post("/youtube_video_transcribe_summarizer", description="The endpoint uses Youtube URL to generate a summary of a video")
async def youtube_video_transcribe_summarizer_gemini(url: str = Form(...)):
    try:
        cache_key = f"youtube_video_transcribe_summarizer:{url}"
        cached_response = redis.get(cache_key)
        if cached_response:
            print("Retrieving response from Redis cache")
            return ResponseText(response=cached_response.decode("utf-8"))

        model = genai.GenerativeModel(settings.GEMINI_FLASH)
        transcript_text = extract_transcript_details(url)
        response = model.generate_content(settings.youtube_transcribe_prompt + transcript_text)
        redis.set(cache_key, response.text, ex=60)
        db = MongoDB()
        payload = {
            "endpoint": "/youtube_video_transcribe_summarizer",
            "url": url,
            "output": response.text
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
    
@app.post("/blog_generator", description="This route will generate the blog based on the desired topic.")
async def blogs(topic: str = Form("Generative AI")):
    try:
        cache_key = f"blog_generator:{topic}"
        cached_response = redis.get(cache_key)
        if cached_response:
            print("Retrieving response from Redis cache")
            return ResponseText(response=cached_response.decode("utf-8"))

        model = genai.GenerativeModel(settings.GEMINI_FLASH)
        blog_prompt = f""" You are expert in blog writing. Write a blog on the topic {topic}. Use a friendly and informative tone, and include examples and tips to encourage readers to get started with the topic provided. """
        response = model.generate_content(blog_prompt)
        redis.set(cache_key, response.text, ex=60)
        db = MongoDB()
        payload = {
            "endpoint": "/blog_generator",
            "topic": topic,
            "output": response.text
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

@app.post("/Text2SQL", description="""This route will generate the SQL query and results from employees table based on the prompt given. \nColumns present in the table are Employee_ID, Name, Department, Title, Email, City, Salary, Work_Experience""")
async def sql_query(prompt: str = Form("Tell me the employees living in city Noida")):
    try:
        cache_key = f"text2sql:{prompt}"
        cached_response = redis.get(cache_key)
        if cached_response:
            print("Retrieving response from Redis cache")
            cached_response = cached_response.decode("utf-8")
            cached_data = json.loads(cached_response)
            return cached_data

        model = genai.GenerativeModel(settings.GEMINI_PRO_1_5)
        response = model.generate_content([settings.text2sql_prompt, prompt])
        output_query = remove_substrings(response.text)
        print(output_query)
        output = read_sql_query(remove_substrings(output_query), settings.EMPLOYEE_DB)
        cached_data = {"response": {"SQL Query": output_query, "Data": output}}
        redis.set(cache_key, json.dumps(cached_data), ex=60)
        db = MongoDB()
        payload = {
            "endpoint": "/Text2SQL",
            "prompt": prompt,
            "SQL Query": output_query,
            "output": output
        }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return {"response": {"SQL Query": output_query, "Data": output}}
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")

@app.post("/questions_generator", description="""The endpoint uses the pdf and generate the questions.
          \nThis will be helpful for the students or teachers preparing for their exams or test. """)
async def pdf_questions_generator(pdf: UploadFile = File(...)):
    try:
        cache_key = f"questions_generator:{pdf.filename}"
        cached_response = redis.get(cache_key)
        if cached_response:
            print("Retrieving response from Redis cache")
            return ResponseText(response=cached_response.decode("utf-8"))

        out = questions_generator(pdf.file)
        redis.set(cache_key, out["output_text"], ex=60)
        db = MongoDB()
        payload = {
            "endpoint": "/questions_generator",
            "output": out["output_text"]
        }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return ResponseText(response=remove_substrings(out["output_text"]))
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")
    
@app.post("/chat_groq", description= """This route uses groq for faster response using Language Processing Unit(LPU).
          \n In model input default is mixtral-8x7b-32768 but you can choose gemma-7b-it, llama3-70b-8192 and llama3-8b-8192.
          \n conversational_memory_length ranges from 1 to 10. It keeps a list of the interactions of the conversation over time.
          It only uses the last K interactions """)
async def groq_chatbot(question: str = Form(...), model: Optional[str] = Form('mixtral-8x7b-32768'), 
    conversational_memory_length: Optional[int] = Form(5)):
    try:
        memory=ConversationBufferWindowMemory(k=conversational_memory_length)
        groq_chat = ChatGroq(groq_api_key= os.environ['GROQ_API_KEY'], model_name=model)
        conversation = ConversationChain(llm=groq_chat,memory=memory)

        response = conversation.invoke(question)
        db = MongoDB()
        payload = {
            "endpoint" : "/chat_groq",
            "question" : question,
            "model" : model,
            "conversational_memory_length": conversational_memory_length,
            "output" : response['response']
            }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        return {"Chatbot": response['response']}
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")


@app.post("/text_summarizer_groq", description= """This route uses groq for faster response using Language Processing Unit(LPU).
          \n This route will provide the concise summary from the text provided & and model used is mixtral-8x7b-32768
           """)
async def groq_text_summary(input_text: str = Form(...)):
    try:
        chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768",api_key=os.environ['GROQ_API_KEY'])
        system = """You are a helpful AI assistant skilled at summarizing text. 
                Your task is to summarize the following text in a clear and concise manner, capturing the main ideas and key points.
                Show result in the points.
            """
        human = "{text}"
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

        chain = prompt | chat
        summary = chain.invoke({"text": input_text})
        summary_text = summary.content
        db = MongoDB()
        payload = {
            "endpoint" : "/text_summarizer_groq",
            "input_text" : input_text,
            "summary" : summary_text
            }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        return {"Summary": summary_text}
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")
    
@app.post("/RAG_PDF_Groq",description="The endpoint uses the pdf and give the answer based on the prompt provided using groq\
          In model input default is mixtral-8x7b-32768 but you can choose gemma-7b-it, llama3-70b-8192 and llama3-8b-8192.")
async def talk_pdf_groq(pdf: UploadFile = File(...),prompt: str = Form(...),
                       model: Optional[str] = Form('llama3-70b-8192')):
    try:
        rag_chain = groq_pdf(pdf.file,model)
        out = rag_chain.invoke(prompt)
        db = MongoDB()
        payload = {
            "endpoint" : "/RAG_PDF_Groq",
            "model" : model,
            "prompt" : prompt,
            "output" : out
        }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return ResponseText(response=out)
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")

@app.post("/summarize_audio", description="""Endpoint to summarize an uploaded audio file using gemini-1.5-pro-latest.""")
async def summarize_audio_endpoint(audio_file: UploadFile = File(...)):
    try:
        cache_key = f"summarize_audio:{audio_file.filename}"
        cached_response = redis.get(cache_key)
        if cached_response:
            print("Retrieving response from Redis cache")
            return ResponseText(response=cached_response.decode("utf-8"))

        summary_text = await summarize_audio(audio_file)
        redis.set(cache_key, summary_text, ex=10)
        db = MongoDB()
        payload = {
            "endpoint": "/summarize_audio",
            "output": summary_text
        }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return ResponseText(response=summary_text)
    except Exception as e:
        return {"error": str(e)}

@app.post("/stream_chat",description="This endpoint streams responses from the language model based on the user's input message.")
async def stream_chat(message: str = Form("What is RLHF in LLM?"),llm: str = Form("llama3-70b-8192")):
    generator = chatbot_send_message(message,model=llm)
    return StreamingResponse(generator, media_type="text/event-stream")

@app.post("/smart_ats", description="""This endpoint is developed using the powerful 
          Gemini Pro 1.5 model to streamline the hiring process by analyzing job descriptions and resumes. 
          It provides valuable insights such as job description match, 
          missing keywords, and profile summary""")
async def ats(resume_pdf: UploadFile = File(...), job_description: str = Form(...)):
    try:
        cache_key = f"smart_ats:{resume_pdf.filename}:{job_description}"
        cached_response = redis.get(cache_key)
        if cached_response:
            print("Retrieving response from Redis cache")
            return ResponseText(response=cached_response.decode("utf-8"))

        text = extraxt_pdf_text(resume_pdf.file)
        model = genai.GenerativeModel(settings.GEMINI_PRO_1_5)
        ats_prompt = f"""
                Hey Act Like a skilled or very experienced ATS (Application Tracking System)
                with a deep understanding of the tech field, software engineering, data science, data analysis,
                and big data engineering. Your task is to evaluate the resume based on the given job description.
                You must consider the job market is very competitive and you should provide 
                the best assistance for improving the resumes. Assign the percentage Matching based 
                on job description and
                the missing keywords with high accuracy
                resume:{text}
                job description:{job_description}

                I want the response as per below structure
                Job Description Match": "%","MissingKeywords": [],"Profile Summary": "".
                Also, tell what more should be added or to be removed in the resume.
                Also, provide the list of some technical questions along with their answers that can be asked in the interview based on the job description.
                """

        response = model.generate_content(ats_prompt)
        redis.set(cache_key, response.text, ex=20)
        db = MongoDB()
        payload = {
            "endpoint": "/smart_ats",
            "resume": text,
            "job description": job_description,
            "ATS Output": response.text
        }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return ResponseText(response=response.text)
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")

@app.post("/advance_rag_llama_index",description="The endpoint build a Router that can choose whether to do vector search or summarization\
          In model input default is llama3-70b-8192 but you can choose mixtral-8x7b-32768, gemma-7b-it and llama3-8b-8192.")
async def llama_index_rag(pdf: UploadFile = File(...),question: str = Form(...),
                       model: Optional[str] = Form('llama3-70b-8192')):
    try:
        rag_output = advance_rag_llama_index(pdf,model,question)
        db = MongoDB()
        payload = {
            "endpoint" : "/advance_rag_llama_index",
            "model" : model,
            "prompt" : question,
            "output" : rag_output
        }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return ResponseText(response=rag_output)
    except Exception as e:
        return ResponseText(response=f"Error: {str(e)}")

@app.post("/text2image",description=
        """
            This API provides access to the following diffusion models for generating images from text prompts.
            
            Models you can use for generating image are:

            1. DreamShaper_v7 - A highly capable and versatile text-to-image model, suitable for a wide range of image generation tasks.

            2. Animagine_xl - A specialized model for generating high-quality anime-style images from text prompts.

            3. Stable_Diffusion_base - The base version of the popular Stable Diffusion model, suitable for general-purpose image generation.

            4. Stable_Diffusion_v2 - The latest version of Stable Diffusion, with improved performance and quality compared to the base version.
        """)
def generate_image(prompt: str = Form("Astronaut riding a horse"), model: str = Form("Stable_Diffusion_base"),
                   token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, os.getenv("TOKEN_SECRET_KEY"), algorithms=[settings.ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        user = users_collection.find_one({"email": email})
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    try:
        if model in settings.diffusion_models:
            def query(payload):
                api_key = os.getenv("HUGGINGFACE_API_KEY")
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.post(settings.diffusion_models[model], headers=headers, json=payload)
                return response.content

            image_bytes = query({"inputs": prompt})
            image = Image.open(io.BytesIO(image_bytes))
            bytes_io = io.BytesIO()
            image.save(bytes_io, format="PNG")
            bytes_io.seek(0)
            return Response(bytes_io.getvalue(), media_type="image/png")
        else:
            return ResponseText(response="Invalid model name")
    # except requests.exceptions.RequestException as e:
    #     print(f"Request Exception: {str(e)}")
    #     return ResponseText(response="Busy server: Please try later")
    except Exception as e:
        return ResponseText(response="Busy server: Please try later")

@app.get("/get_data/{endpoint_name}")
async def get_data(endpoint_name: str, token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, os.getenv("TOKEN_SECRET_KEY"), algorithms=[settings.ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        user = users_collection.find_one({"email": email})
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    cache_key = f"{endpoint_name}"
    cached_data = redis.get(cache_key)

    if cached_data:
        print("Retrieving data from Redis cache")
        data = json.loads(cached_data)
        return data
    
    print("Retrieving data from MongoDB")
    db = MongoDB()
    data = db.read_by_endpoint(endpoint_name)

    if isinstance(data, list):
        redis.set(cache_key, json.dumps(data), ex=60)

    return data

# @app.post("/news_agent",description="""
#           This endpoint leverages AI agents to conduct research and generate articles on various tech topics. 
#           The agents are designed to uncover groundbreaking technologies and narrate compelling tech stories
#           """)
# async def run_news_agent(topic: str = Form("AI in healthcare")):
#     try:
#         cache_key = f"news_agent:{topic}"
#         cached_response = redis.get(cache_key)
#         if cached_response:
#             print("Retrieving response from Redis cache")
#             return ResponseText(response=cached_response.decode("utf-8"))

#         output = run_crew(topic=topic)
#         redis.set(cache_key, output, ex=10)
#         db = MongoDB()
#         payload = {
#             "endpoint": "/news_agent",
#             "topic" : topic,
#             "output": output
#         }
#         mongo_data = {"Document": payload}
#         result = db.insert_data(mongo_data)
#         print(result)
#         return ResponseText(response=output)
#     except Exception as e:
#         return {"error": str(e)}

@app.post("/query_db",description="""
          The Query Database endpoint provides a service for interacting with SQL databases using a Cohere ReAct Agent. 
          It leverages Langchain's existing SQLDBToolkit to answer questions and perform queries over SQL database.
          """)
async def query_db(database: UploadFile = File(...), prompt: str = Form(...)):
    try: 
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + database.filename.split('.')[-1]) as temp_file:
            shutil.copyfileobj(database.file, temp_file)
            db_path = temp_file.name

        llm = ChatCohere(model="command-r-plus", temperature=0.1, verbose=True,cohere_api_key=os.getenv("COHERE_API_KEY"))
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        context = toolkit.get_context()
        tools = toolkit.get_tools()
        chat_prompt = ChatPromptTemplate.from_template("{input}")

        agent = create_cohere_react_agent(
            llm=llm,
            tools=tools,
            prompt=chat_prompt
        )
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            return_intermediate_steps=False,
        )

        preamble = settings.QUERY_DB_PROMPT.format(schema_info=context)
        
        out = agent_executor.invoke({
           "input": prompt,
           "preamble": preamble
        })

        output  = parse_sql_response(out["output"])
        db = MongoDB()
        payload = {
            "endpoint": "/query_db",
            "input": prompt,
            "output": output
        }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)

        return ResponseText(response=output)

    except Exception as e:
        raise Exception(f"Error handling uploaded file: {e}")

    finally:
        database.file.close()

@app.post("/MediGem",description="Medical Diagnosis AI Assistant")
async def medigem(image_file: UploadFile = File(...)):
    
    image = image_file.file.read()
    image_parts = [{
            "mime_type": "image/jpeg",
            "data": image
        }]
    model = genai.GenerativeModel(settings.GEMINI_PRO_1_5)
    response = model.generate_content([image_parts[0], settings.MEDI_GEM_PROMPT])
    db = MongoDB()
    payload = {
            "endpoint" : "/MediGem",
            "output" : response.text
        }
    mongo_data = {"Document": payload}
    result = db.insert_data(mongo_data)
    print(result)
    return ResponseText(response=remove_substrings(response.text))

@app.post("/NoteGem", description="This API endpoint leverages the Google Gemini AI Model to generate comprehensive notes from YouTube video transcripts")
@limiter.limit("5/2minute")

async def process_video(request: Request, video_url: str = Form(...)):
    video_id = extract_video_id(video_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    try:
        transcript_text = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([i["text"] for i in transcript_text])
    except (TranscriptsDisabled, NoTranscriptFound):
        return {"transcript": "Transcript not available", "error": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        cache_key = f"notegem:{video_id}"
        cached_response = redis.get(cache_key)
        if cached_response:
            print("Retrieving response from Redis cache")
            return ResponseText(response=cached_response.decode("utf-8"))
        
        model = genai.GenerativeModel(settings.GEMINI_PRO_1_5)
        response = model.generate_content(settings.NOTE_GEN_PROMPT + transcript)
        summary = response.text
        redis.set(cache_key, summary, ex=60)
        db = MongoDB()
        payload = {
                "endpoint" : "/NoteGem",
                "video_url": video_url,
                "output" : response.text
            }
        mongo_data = {"Document": payload}
        result = db.insert_data(mongo_data)
        print(result)
        return ResponseText(response=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)