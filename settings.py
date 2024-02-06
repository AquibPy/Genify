PALM_MODEL = "models/text-bison-001"
FAQ_FILE = 'data\\faqs.csv'
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

