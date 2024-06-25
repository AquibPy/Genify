[![wakatime](https://wakatime.com/badge/user/36345258-0e7d-4799-92b8-d71688c8e385/project/018d63c9-7b84-4706-8a54-cc5a26d06bca.svg)](https://wakatime.com/badge/user/36345258-0e7d-4799-92b8-d71688c8e385/project/018d63c9-7b84-4706-8a54-cc5a26d06bca)

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

## DataBase Support

This project supports MongoDB and Redis for data storage and caching respectively.

### MongoDB

![MongoDB Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/MongoDB_Logo.svg/512px-MongoDB_Logo.svg.png)

- **Description:** MongoDB is a popular NoSQL database used for storing unstructured data.
- **Scalability:** Highly scalable, suitable for handling large volumes of data.
- **Flexibility:** Offers flexible schemas, allowing for dynamic and evolving data structures.
- **Document-Oriented:** Stores data in JSON-like documents, making it easy to work with for developers.
- **High Performance:** Designed for high performance and low latency, suitable for real-time applications.
- **Community Support:** Large and active community, providing resources, documentation, and support.
- **Cross-Platform:** Supports multiple platforms including Windows, macOS, and Linux.
- **Integration:** Easily integrates with various programming languages and frameworks.
- **Aggregation Framework:** Provides powerful aggregation framework for data analysis and reporting.
- **Replication and Sharding:** Supports replication and sharding for high availability and scalability.

### Redis

![Redis Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/Logo-redis.svg/2560px-Logo-redis.svg.png)

- **Description:** Redis is an open-source, in-memory data structure store used as a database, cache, and message broker.
- **In-Memory Storage:** Data is stored in memory, allowing for fast read and write operations.
- **Data Structures:** Supports various data structures such as strings, hashes, lists, sets, and sorted sets.
- **High Performance:** Known for its high performance and low latency, making it suitable for real-time applications.
- **Persistence Options:** Provides different persistence options including snapshots and append-only files for data durability.
- **Pub/Sub Messaging:** Includes pub/sub messaging functionality, allowing for message passing between clients.
- **Replication:** Supports master-slave replication for high availability and fault tolerance.
- **Lua Scripting:** Allows for scripting using Lua, enabling complex operations and transactions.
- **Clustering:** Supports clustering for horizontal scaling and distributing data across multiple nodes.
- **Atomic Operations:** Provides atomic operations on data structures, ensuring consistency and reliability.

# Rate Limiting

This FastAPI application includes rate limiting to control the number of requests that can be made to certain endpoints within a specified time frame. Rate limiting helps prevent abuse of the API and ensures fair usage among consumers.

## Rate Limiting Configuration

Rate limiting is implemented using `slowapi`, which provides middleware for rate limiting based on IP address or other identifiers.

### Configuration Details

- **Limits**: Requests are limited to a certain number per minute.
- **Identifier**: Rate limiting is applied based on the IP address of the client.
- **Exceeding Limit**: Clients exceeding the limit receive a 429 HTTP status code with an appropriate message.

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

### 19. Text to Image using Diffusion Models

- **Route:** `/text2image`
- **Description:** This route allows you to generate images using various diffusion models available on Hugging Face.

    You can choose from the following models -

    **DreamShaper v7**: A highly capable and versatile text-to-image model, suitable for a wide range of image generation tasks.

    **Animagine XL**: A specialized model for generating high-quality anime-style images from text prompts.

    **Stable Diffusion Base**: The base version of the popular Stable Diffusion model, suitable for general-purpose image generation.

    **Stable Diffusion v2**: The latest version of Stable Diffusion, with improved performance and quality compared to the base version.

    To generate an image, send a POST request to the `/text2image` endpoint with the desired model and prompt in the request body.

    **Request Body:** `{ "model": "DreamShaper_v7", "prompt": "An astronaut riding a horse on the moon" }` The response will be the generated image in PNG format.

### 20. Advanced RAG with LLaMA Index

- **Route:** `/advance_rag_llama_index`
- **Description:** This API provides an advanced retrieval-augmented generation (RAG) functionality using the LLaMA Index. It allows users to upload a document and ask questions, enabling the API to search for specific facts or summarize the document based on the query.
- **Feature:**
  - Upload PDF documents and ask natural language questions.
  - Choose between vector search or summarization for responses.
  - Support for multiple Open Source LLMs models.

### 21. Tech News Agent using Crew ai

- **Route:** `/news_agent`
- **Description:** This endpoint leverages AI agents to conduct research and generate articles on various tech topics. The agents are designed to uncover groundbreaking technologies and narrate compelling tech stories.
- **Features:**
  - Accepts a `topic` parameter specifying the tech topic of interest.
  - Utilizes caching mechanisms for improved performance by storing and retrieving responses from Redis cache.
  - Integrates with MongoDB for storing endpoint usage data.
  - Returns articles and research findings related to the specified tech topic.
  - Handles exceptions gracefully and returns error messages in JSON format.

### 22. Query Database

- **Route:** `/query_db`
- **Description:** This API endpoint facilitates querying SQL databases using a Cohere ReAct Agent, integrated with Langchain's SQLDBToolkit.
- **Feature:**
  - Upload a `.db` file containing the database to be queried.
  - Provide a natural language prompt or query to retrieve relevant information
  from the database.
  - Utilizes a Cohere ReAct Agent to process user queries and generate responses.

### 23. MediGem: Medical Diagnosis AI Assistant

- **Route:** `/MediGem`
- **Description:** This API endpoint leverages the New Google Gemini AI Model to analyze medical images and identify potential health conditions.
- **Feature:**
  - **Upload a Medical Image:** Users can upload a medical image in JPEG format for analysis.
  - **AI Analysis:** The AI model examines the image to identify anomalies, diseases, or health issues.
  - **Findings Report:** Documents observed anomalies or signs of disease in a structured format.
  - **Recommendations:** Suggests potential next steps, including further tests or treatments.
  - **Treatment Suggestions:** Recommends possible treatment options or interventions if applicable.
  - **Image Quality Note:** Indicates if certain aspects are 'Unable to be determined based on the provided image.'
  - **Disclaimer:** Includes the disclaimer: "Consult with a Doctor before making any decisions."

### 24. NoteGem: Automated Note-Taking Assistant

- **Route:** `/NoteGem`
- **Description:** This API endpoint leverages the Google Gemini AI Model to generate comprehensive notes from YouTube video transcripts.
- **Feature:**
  - **Input Video URL:** Users can provide a YouTube video URL for processing.
  - **Transcript Extraction:** The API extracts the transcript from the provided YouTube video.
  - **Error Handling for Transcripts:** If the transcript is not available, it returns a message indicating that the transcript is not available for transcription.
  - **AI Summary Generation:** The AI model generates a structured summary of the transcript focusing on main points, critical information, key takeaways, examples or case studies, quotes, and actionable steps.

### 25. Investment Risk Analyst Agent

- **Route:** `/investment_risk_agent`
- **Description:** This API endpoint coordinates a team of AI agents to perform comprehensive investment risk analysis and strategy development.
- **Feature:**
  - **Input Data:** Users can provide input data including stock selection, initial capital, risk tolerance, trading strategy preference, and news impact consideration.
  - **Data Analysis:** The data analyst agent processes the input data to extract relevant financial information.
  - **Strategy Development:** The trading strategy agent formulates a suitable trading strategy based on the analyzed data and user preferences.
  - **Risk Assessment:** The risk management agent evaluates potential risks associated with the trading strategy and suggests mitigation measures.
  - **Execution Planning:** The execution agent develops a detailed plan for executing the trading strategy, considering the assessed risks.

## Usage

Each endpoint accepts specific parameters as described in the respective endpoint documentation. Users can make POST requests to these endpoints with the required parameters to perform the desired tasks.

## Dependencies

- FastAPI: A modern, fast (high-performance) web framework for building APIs with Python.
- Pydantic: Data validation and settings management using Python type annotations.
- Other dependencies specific to individual endpoint functionalities.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AquibPy/Genify.git
   ```

2. Install dependencies:

    ```python
    pip install -r requirements.txt
    ```

3. Create ```.env``` file

    Save Google and Hugging Face API key in this file.

## Running the Server

Start the FastAPI server by running the following command:
    ```
    fastapi run api.py
    ```

## Contributing

Contributions are welcome! Feel free to open a pull request or submit an issue for any bugs or feature requests.
