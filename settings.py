GEMINI_PRO_1_5 = "gemini-1.5-pro-latest"
GEMINI_FLASH = "gemini-1.5-flash-latest"
GEMINI_FLASH_8B = "gemini-1.5-flash-8b"
GOOGLE_EMBEDDING = "models/embedding-001"
FAQ_FILE = 'data/faqs.csv'
EMPLOYEE_DB = "data/employees.db"
INSTRUCTOR_EMBEDDING = "sentence-transformers/all-MiniLM-l6-v2"
VECTORDB_PATH = "faiss_index"
REDIS_PORT = 18804
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
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

summary_para = """
In the vast landscape of human history, civilizations have risen and fallen, leaving behind legacies that shape our present. 
From the ancient civilizations of Mesopotamia and Egypt, where the foundations of writing, agriculture, and governance were laid, 
to the grand empires of Rome and China, which expanded their reach through conquest and trade, the story of humanity is one of ambition, innovation, and conflict. 
The Middle Ages saw the emergence of feudalism in Europe, characterized by the exchange of land for loyalty and protection, 
while the Islamic Golden Age ushered in a period of scientific, artistic, and philosophical advancement in the Muslim world. 
The Renaissance in Europe sparked a revival of classical learning and ushered in an era of exploration and discovery, 
leading to the age of Enlightenment, where reason and empiricism challenged traditional authority. 
The Industrial Revolution transformed societies with technological advancements, urbanization, 
and shifts in economic production, while the 20th century witnessed unprecedented global conflicts, 
technological leaps, and social revolutions. Today, in the 21st century, 
we stand at the intersection of unprecedented technological advancement and pressing global challenges, 
navigating issues of climate change, political polarization, and the ethical implications of artificial intelligence.
As we reflect on the journey of humanity, from ancient civilizations to the digital age, we are reminded of our shared past and
the collective responsibility to shape a more equitable and sustainable future.
"""

QUERY_DB_PROMPT = """## Task And Context
        You use your advanced complex reasoning capabilities to help people by answering their questions and other requests interactively. 
        You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you,
        which you use to research your answer. You may need to use multiple tools in parallel or sequentially to complete your task. 
        You should focus on serving the user's needs as best you can, which will be wide-ranging.

        ## Style Guide
        Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.

        ## Additional Information
        You are an expert who answers the user's question by creating SQL queries and executing them.
        You are equipped with a number of relevant SQL tools.
        You should also present the SQL query used to provide the data.

        Here is information about the database:
        {schema_info}
        """

MEDI_GEM_PROMPT = """
    As a highly skilled medical practitioner specializing in image analysis, you are tasked with examining medical images for a renowned hospital.
    Your expertise is crucial in identifying any anomalies,diseases, or health issues that may be present in the image.
    
    Your Responsibilities include:

    1. Detailed Analysis: Thoroughly analyze each image, focusing on identifying any abnormal findings.
    2. Findings Report: Document all the observed anomalies or signs of disease. Clearly articulate these findings in structured format.
    3. Recommendations and Next Steps: Based on your analysis, suggest potential next steps. including further tests or treatments as applicable.
    4. Treatment Suggestions: If appropriate, recommend possible treatment options or interventions.
    
    Important Notes:

    1. Scope of Response: Only respond if the image pertains to human health issues.
    2. Clarity of Image: In cases where the image quality impedes clear analysis, note that certain aspects are 'Unable to be determined based on the provided image.'
    3. Disclaimer: Accompany your analysis with the disclaimer: "Consult with a Doctor before making any decisions."
    4.Your insights are invaluable in guiding clinical decisions. Please proceed with analysis, adhering to the structured approach outlined above.

    Please provide me an output with these 4 headings Detailed Analysis, Findings Report, Recommendations and Next Steps and Treatment Suggestions.

    """

NOTE_GEN_PROMPT = """
You are a professional note-taker with expertise in distilling key insights from video content. Your task is to generate a comprehensive, yet concise set of notes from the provided video transcript. Focus on the following:

1. Main points
2. Critical information
3. Key takeaways
4. Examples or case studies
5. Quotes or important statements
6. Actionable steps or recommendations

Make sure the notes are well-structured and formatted as bullet points. The total length should not exceed 1000 words. Please summarize the following text:
"""

AGRILENS_DEFAULT_PROMPT = """
Analyze the uploaded image, which is related to agriculture.
Identify the key elements present in the image, such as crops, soil, pests, diseases, farming equipment, or other agricultural features.
Provide the following details:
1. Description: A brief overview of what is visible in the image.
2. Detailed Identification: Name the type of crop, pest, disease, or farming equipment, if applicable.
3. Condition Assessment: If the image contains crops, assess their health (e.g., healthy, stressed, diseased). If there are visible pests or diseases, specify their type and possible impact.
4. Recommendations: Suggest actionable insights to improve the situation, such as applying fertilizers, pesticides, irrigation, or other agricultural practices.
5. Additional Observations: Any other insights or anomalies you notice in the image relevant to agriculture.
"""