from crewai_tools import ScrapeWebsiteTool, SerperDevTool
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()