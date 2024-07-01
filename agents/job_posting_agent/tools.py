from crewai_tools import SerperDevTool, WebsiteSearchTool
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

serper_search_tool = SerperDevTool()
web_search_tool =  WebsiteSearchTool(
    config=dict(
        llm=dict(
            provider="google", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="gemini-1.5-flash-latest",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)