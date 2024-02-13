import os
from settings import PALM_MODEL,FAQ_FILE,INSTRUCTOR_EMBEDDING,VECTORDB_PATH,qa_prompt,prompt_pdf
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from PyPDF2 import PdfReader

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

llm = GoogleGenerativeAI(model=PALM_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY"),temperature=0)

PaLM_embeddings = GooglePalmEmbeddings(google_api_key=os.getenv("GOOGLE_API_KEY"))

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
    llm = GoogleGenerativeAI(model=PALM_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY"),temperature=0.7)
    vectordb = FAISS.load_local(VECTORDB_PATH,PaLM_embeddings)
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
    llm = GoogleGenerativeAI(model=PALM_MODEL, google_api_key=os.getenv("GOOGLE_API_KEY"),temperature=0.9)
    if url:
        loader = UnstructuredURLLoader(urls=url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
        docs = text_splitter.split_documents(data)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20,length_function=len,is_separator_regex=False)
        docs = text_splitter.create_documents(doc)

    vectorstore = FAISS.from_documents(documents = docs,embedding=PaLM_embeddings)
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
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

    llm = GoogleGenerativeAI(model="gemini-pro",temperature=0.3)
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

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain.invoke("Do you have javascript course?"))