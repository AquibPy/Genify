# Genify

This project contains a collection of APIs that leverage Generative AI models to perform various tasks. These APIs are designed to address different use cases such as extracting information from invoices, generating responses from FAQs, summarizing YouTube videos, generating blog posts, analyzing health-related data from images, and executing SQL queries from text prompts for database operations related to employee data.

## Table of Contents

- [Overview](#overview)
- [Endpoints](#endpoints)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [Contributing](#contributing)

## Overview

Generative AI, powered by advanced machine learning models, enables the creation of content, responses, and insights that mimic human-like behavior. This project harnesses the capabilities of Generative AI to provide a set of APIs catering to various tasks across different domains.

## Endpoints

### 1. Invoice Extractor

- **Route:** `/invoice_extractor`
- **Description:** This endpoint extracts information from invoices based on provided images and prompts. It utilizes a Generative AI model to process the input and generate relevant data.

### 2. QA from FAQs

- **Route:** `/qa_from_faqs`
- **Description:** The QA from FAQs endpoint uses retrieved question-answer pairs to generate responses to user prompts. It employs a Generative AI model trained on FAQ datasets to provide accurate and relevant answers.

### 3. QA URL/Doc

- **Route:** `/qa_url_doc`
- **Description:** With this endpoint, users can provide either a URL or upload a document (e.g., news article, blog) along with a question. The endpoint generates responses to the question based on the content of the provided URL or document using Generative AI models.

### 4. YouTube Video Transcribe Summarizer

- **Route:** `/youtube_video_transcribe_summarizer`
- **Description:** This endpoint utilizes a Generative AI model to summarize YouTube videos by transcribing the video content and generating a concise summary.

### 5. Nutritionist Expert

- **Route:** `/health_app_gemini`
- **Description:** The Nutritionist Expert endpoint analyzes images of food items to extract nutritional information and provide insights on calorie intake and weight management. Users can input their height and weight for personalized recommendations.

### 6. Blog Generator

- **Route:** `/blog_generator`
- **Description:** Generate informative and engaging blog posts on a specified topic using a Generative AI model. The endpoint generates content in a friendly and informative tone, encouraging readers to explore the topic further.

### 7. Talk2PDF

- **Route:** `/talk2PDF`
- **Description:** Extract information from PDF documents and provide responses based on user prompts. The endpoint utilizes Generative AI models to process the document content and generate relevant answers.

### 8. Text2SQL

- **Route:** `/Text2SQL`
- **Description:** Generate SQL queries and results from an employee database based on user prompts. The endpoint uses Generative AI models to convert text prompts into SQL queries for database operations.

### 9. Questions Generator

- **Route:** `/questions_generator`
- **Description:** The endpoint uses the pdf and generate the questions.It will be helpful for the students or teachers preparing for their exams or test.

### 10. ChatBot Using Groq

- **Route:** `/chat_groq`
- **Description:** This route utilizes Groq for enhanced language processing speed, with a default model input of mixtral-8x7b-32768, but offering alternatives like llama2-70b-4096 and gemma-7b-it, and a conversational memory length option of 1 to 10, which maintains a list of recent interactions in the conversation, considering only the latest K interactions.

### 11. Text Summarizer

- **Route:** `/text_summarizer_groq`
- **Description:**  Dive into a realm of creativity with our text summarization endpoint, where the model mixtral-8x7b-32768 crafts concise summaries from your input text, delivering insights at the speed of thought.

### 12. RAG Using Groq

- **Route:** `/RAG_PDF_Groq`
- **Description:** This endpoint uses the pdf and give the answer based on the prompt provided using Groq,with a default model input of llama2-70b-4096, but offering alternatives like mixtral-8x7b-32768 and gemma-7b-it.

### 13. Audio Summarizer

- **Route:** `/summarize_audio`
- **Description:** Endpoint to summarize an uploaded audio file using gemini-1.5-pro-latest.

### 14. Chat Streaming

- **Route:** `/stream_chat`
- **Description:** This endpoint streams responses from the language model based on the user's input message.

### 15. ChatBot

- **Route:** `/chatbot`
- **Description:** Provides a simple web interface to interact with the chatbot.
- **Try ChatBot:**  [Talk to LLAMA 3](https://llm-pgc4.onrender.com/chatbot)

### 16. Resume Evaluator

- **Route:** `/smart_ats`
- **Description:** The Resume Evaluator endpoint allows users to evaluate resumes against provided job descriptions. It leverages a Generative AI model to analyze resume content and generate insights such as job description match
percentage, missing keywords, and profile summary.
- **Input:** Users need to upload a resume file and provide a job description as text.
- **Output:** The endpoint provides insights in JSON format, including job description match.

### 17. Blog Genertor UI

- **Route:** `/blog_generator_ui`
- **Description:** Provides a simple web interface to interact with the Blog Generator.
- **Try Blog Generator:**  [Blog Generator](https://llm-pgc4.onrender.com/blog_generator_ui)

### 18. Smart ATS UI

- **Route:** `/ats`
- **Description:** Provides a simple web interface to interact with the smart ats.
- **Try ATS:**  [Smart ATS](https://llm-pgc4.onrender.com/blog_generator_ui)

## Usage

Each endpoint accepts specific parameters as described in the respective endpoint documentation. Users can make POST requests to these endpoints with the required parameters to perform the desired tasks.

## Dependencies

- FastAPI: A modern, fast (high-performance) web framework for building APIs with Python.
- Pydantic: Data validation and settings management using Python type annotations.
- Other dependencies specific to individual endpoint functionalities.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AquibPy/LLM-use-cases-API.git
   ```

2. Install dependencies:

    ```python
    pip install -r requirements.txt
    ```

3. Create ```.evn``` file

    Save Google and Hugging Face API key in this file.

## Running the Server

Start the FastAPI server by running the following command:
    ```
    uvicorn main:app --reload
    ```

## Contributing

Contributions are welcome! Feel free to open a pull request or submit an issue for any bugs or feature requests.
