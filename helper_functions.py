import os
from settings import GOOGLE_EMBEDDING,FAQ_FILE,INSTRUCTOR_EMBEDDING,VECTORDB_PATH,qa_prompt,\
prompt_pdf,question_prompt_template,question_refine_template, GEMINI_PRO
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredURLLoader,PyPDFLoader,WebBaseLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from PyPDF2 import PdfReader
import sqlite3
from langchain_community.embeddings import GooglePalmEmbeddings
import tempfile
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
from langchain.schema import HumanMessage

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


PaLM_embeddings = GooglePalmEmbeddings(google_api_key=os.getenv("GOOGLE_API_KEY"))

google_embedding = GoogleGenerativeAIEmbeddings(model = GOOGLE_EMBEDDING)

'''
if you want you can try instructor embeddings also. Below is thge code :

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HUGGINGFACE_API_KEY"), model_name=INSTRUCTOR_EMBEDDING,query_instruction="Represent the query for retrieval: "
)
'''

def get_gemini_response(input, image_file, prompt):
    try:
        print(prompt,input)
        model = genai.GenerativeModel("gemini-pro-vision")
    
        response = model.generate_content([input, image_file[0], prompt])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def get_gemini_response_health(image_file, prompt):
    try:
        model = genai.GenerativeModel("gemini-pro-vision")
    
        response = model.generate_content([image_file[0], prompt])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def create_vector_db():
    loader = CSVLoader(file_path=FAQ_FILE)
    data = loader.load()
    vectordb = FAISS.from_documents(documents = data,embedding=PaLM_embeddings)
    vectordb.save_local(VECTORDB_PATH)

def get_qa_chain():
    llm = GoogleGenerativeAI(model= GEMINI_PRO, google_api_key=os.getenv("GOOGLE_API_KEY"),temperature=0.7)
    vectordb = FAISS.load_local(VECTORDB_PATH,PaLM_embeddings,allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    PROMPT = PromptTemplate(
        template=qa_prompt, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

def get_url_doc_qa(url,doc):
    llm = GoogleGenerativeAI(model= GEMINI_PRO, google_api_key=os.getenv("GOOGLE_API_KEY"),temperature=0.3)
    if url:
        loader = WebBaseLoader(url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000,
                chunk_overlap = 200
            )
        docs = text_splitter.split_documents(data)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,length_function=len,is_separator_regex=False)
        docs = text_splitter.create_documents(doc)

    vectorstore = FAISS.from_documents(documents = docs,embedding=google_embedding)
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=vectorstore.as_retriever(),
                                        input_key="query",
                                        return_source_documents=True)
    return chain

def extract_transcript_details(youtube_video_url):
    try:
        video_id=youtube_video_url.split("=")[1]
        
        transcript_text=YouTubeTranscriptApi.get_transcript(video_id)

        transcript = ""
        for i in transcript_text:
            transcript += " " + i["text"]

        return transcript

    except Exception as e:
        raise e

def get_gemini_pdf(pdf):
    text = "".join(page.extract_text() for page in PdfReader(pdf).pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embedding=google_embedding)
    llm = GoogleGenerativeAI(model= GEMINI_PRO, google_api_key=os.getenv("GOOGLE_API_KEY"),temperature=0.7)
    retriever = vector_store.as_retriever(score_threshold=0.7)
    PROMPT = PromptTemplate(
        template=prompt_pdf, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

def read_sql_query(query,db):
    conn=sqlite3.connect(db)
    cur=conn.cursor()
    cur.execute(query)
    rows=cur.fetchall()
    conn.commit()
    conn.close()
    for row in rows:
        print(row)
    return rows

def remove_substrings(input_string):
    modified_string = input_string.replace("/n", "")
    modified_string = modified_string.replace("/", "")
    return modified_string


def questions_generator(doc):
    # loader = PdfReader(doc)
    # data = loader.load()

    question_gen =  "".join(page.extract_text() for page in PdfReader(doc).pages)

    # for page in data:
    #     question_gen += page.page_content
        
    splitter_ques_gen = TokenTextSplitter(
        chunk_size = 10000,
        chunk_overlap = 200
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    # splitter_ans_gen = TokenTextSplitter(chunk_size = 1000,chunk_overlap = 100)
    # document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    llm_ques_gen_pipeline = ChatGoogleGenerativeAI(model= GEMINI_PRO,google_api_key=os.getenv("GOOGLE_API_KEY"),temperature=0.3)
    PROMPT_QUESTIONS = PromptTemplate(template=question_prompt_template, input_variables=["text"])
    REFINE_PROMPT_QUESTIONS = PromptTemplate(input_variables=["existing_answer", "text"],template=question_refine_template)
    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline, 
                                            chain_type = "refine", 
                                            verbose = True, 
                                            question_prompt=PROMPT_QUESTIONS, 
                                            refine_prompt=REFINE_PROMPT_QUESTIONS)

    ques = ques_gen_chain.run(document_ques_gen)
    return ques

def groq_pdf(pdf,model):
    llm = ChatGroq(
            api_key=os.environ['GROQ_API_KEY'],
            model_name=model
    )
    text = "".join(page.extract_text() for page in PdfReader(pdf).pages)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embedding=google_embedding)
    retriever = vectorstore.as_retriever()
    rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

async def summarize_audio(audio_file):
    """Summarize the audio using Google's Generative API."""
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

    # Save the audio file to a temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.'+audio_file.filename.split('.')[-1]) as tmp_file:
            tmp_file.write(await audio_file.read())
            audio_file_path = tmp_file.name
    except Exception as e:
        raise Exception(f"Error handling uploaded file: {e}")

    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(
        [
            "Please summarize the following audio.",
            audio_file
        ]
    )

    return response.text


async def chatbot_send_message(content: str,model: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatGroq(
        temperature=0, 
        groq_api_key=os.environ['GROQ_API_KEY'], 
        model_name=model,
        streaming=True,
        verbose=True,
        callbacks=[callback],
    )

    task = asyncio.create_task(
        model.agenerate(messages=[[HumanMessage(content=content)]])
    )

    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task

def extraxt_pdf_text(uploaded_file):
    reader=PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain.invoke("Do you have javascript course?"))