from crewai import Agent
from .tools import search_tool,scrape_tool
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import settings

llm=ChatGoogleGenerativeAI(model=settings.GEMINI_FLASH,
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))

data_analyst_agent = Agent(
    role="Data Analyst",
    goal="Monitor and analyze market data in real-time to identify trends and predict market movements.",
    backstory="""
    Specializing in financial markets, this agent employs statistical modeling and machine learning techniques to provide critical insights. 
    Renowned for its proficiency in data analysis, the Data Analyst Agent serves as a pivotal resource for informing trading decisions.
    """,
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    llm = llm
)

trading_strategy_agent = Agent(
    role="Trading Strategy Developer",
    goal="Develop and test various trading strategies leveraging insights from the Data Analyst Agent.",
    backstory="""
    Possessing a deep understanding of financial markets and quantitative analysis, this agent formulates and optimizes trading strategies. 
    It assesses the performance of diverse approaches to identify the most profitable and risk-averse options.
    """,
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    llm = llm
)

execution_agent = Agent(
    role="Trade Advisor",
    goal="Recommend optimal trade execution strategies based on approved trading plans.",
    backstory="""
    Specializing in the analysis of timing, price, and logistical details of potential trades, 
    this agent evaluates these factors to provide well-founded recommendations. 
    Its expertise ensures that trades are executed efficiently and in alignment with the overall strategy.""",
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    llm = llm
)

risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate and provide insights on the risks associated with potential trading activities.",
    backstory="""
    With extensive expertise in risk assessment models and market dynamics, 
    this agent thoroughly examines the potential risks of proposed trades. 
    It delivers comprehensive analyses of risk exposure and recommends safeguards to ensure that trading activities align with the firm's risk tolerance.
    """,
    verbose=True,
    allow_delegation=True,
    tools=[scrape_tool, search_tool],
    llm = llm
)