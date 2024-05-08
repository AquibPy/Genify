GEMINI_PRO = "gemini-pro"
GEMINI_PRO_1_5 = "gemini-1.5-pro-latest"
GOOGLE_EMBEDDING = "models/embedding-001"
FAQ_FILE = 'data\\faqs.csv'
EMPLOYEE_DB = "data/employees.db"
INSTRUCTOR_EMBEDDING = "sentence-transformers/all-MiniLM-l6-v2"
VECTORDB_PATH = "faiss_index"
qa_prompt = """
    Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}
    """
invoice_prompt = """
               You are an expert in understanding invoices.
               You will receive input images as invoices &
               you will have to answer questions based on the input image
               """
youtube_transcribe_prompt = """You are Yotube video summarizer. You will be taking the transcript text
                and summarizing the entire video and providing the important summary in points
                within 250 words. Please provide the summary of the text given here:  """

prompt_pdf = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "Answer is not available in the context !!!!!!!", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

text2sql_prompt = """
    You are an expert in converting English questions to SQL query!
    The SQL database has the name EMPLOYEES and has the following columns - Employee_ID, Name, 
    Department, Title, Email, City, Salary, 
    Work_Experience \n\nFor example,\nExample 1 - How many entries of records are present?, 
    the SQL command will be something like this SELECT COUNT(*) FROM EMPLOYEES ;
    \nExample 2 - Tell me all the employees living in Noida city?, 
    the SQL command will be something like this SELECT * FROM EMPLOYEES 
    where City="Noida"; 
    also the sql code should not have ``` in beginning or end and sql word in output
    """

question_prompt_template = """
You are an expert at creating questions based on study materials and reference guide.
Your goal is to prepare a student or teacher for their exam and tests.
You do this by asking questions about the text below:

------------
{text}
------------

Create questions that will prepare the student or teacher for their tests.
Make sure not to lose any important information.

QUESTIONS:
"""

question_refine_template = ("""
You are an expert at creating practice questions based on study materials and reference guide.
Your goal is to help a student or teacher prepare for their exam or test.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.
QUESTIONS:
"""
)

diffusion_models = {
    "DreamShaper_v7"  : "https://api-inference.huggingface.co/models/SimianLuo/LCM_Dreamshaper_v7",
    "Animagine_xl" : "https://api-inference.huggingface.co/models/cagliostrolab/animagine-xl-3.0",
    "Stable_Diffusion_base" : "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
    "Stable_Diffusion_v2" : "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
}