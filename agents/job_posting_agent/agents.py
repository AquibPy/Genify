from crewai import Agent
from .tools import web_search_tool, serper_search_tool
import os
import settings

os.environ["GEMINI_API_KEY"] = os.environ.get('GEMINI_API_KEY')

llm = "gemini/gemini-1.5-flash-8b"


class JobAgents():
    def research_agent(self):
        return Agent(
            role = "Research Analyst",
            goal = "Analyze the company website and provided description to extract insights on culture, values, and specific needs.",
            tools = [web_search_tool,serper_search_tool],
            backstory = "Expert in analyzing company cultures and identifying key values and needs from various sources, including websites and brief descriptions.",
            llm = llm,
            verbose = True,
            allow_delegation=False
        )
    
    def writer_agent(self):
        return Agent(
            role='Job Description Writer',
			goal='Use insights from the Research Analyst to create a detailed, engaging, and enticing job posting.',
			tools= [web_search_tool,serper_search_tool],
			backstory='Skilled in crafting compelling job descriptions that resonate with the company\'s values and attract the right candidates.',
            llm = llm,
			verbose=True,
            allow_delegation=False
        )
    
    def review_agent(self):
        return Agent(
			role='Review and Editing Specialist',
			goal='Review the job posting for clarity, engagement, grammatical accuracy, and alignment with company values and refine it to ensure perfection.',
			tools= [web_search_tool,serper_search_tool],
			backstory='A meticulous editor with an eye for detail, ensuring every piece of content is clear, engaging, and grammatically perfect.',
            llm = llm,
			verbose=True,
            allow_delegation=False
			)