from crewai import Crew, Process
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
# import settings
import os
from .agents import data_analyst_agent,trading_strategy_agent,execution_agent,risk_management_agent
from .tasks import data_analysis_task,strategy_development_task,risk_assessment_task,execution_planning_task

llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",
                           verbose=True,
                           temperature=0.7,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))


financial_trading_crew = Crew(
    agents=[data_analyst_agent,
            trading_strategy_agent,
            execution_agent,
            risk_management_agent],

    tasks=[data_analysis_task,
           strategy_development_task,
           execution_planning_task,
           risk_assessment_task],

    manager_llm=llm,
    process=Process.hierarchical,
    verbose=True
)

def run_investment_crew(input_data):
    result = financial_trading_crew.kickoff(inputs=input_data)
    return result

if __name__=='__main__':
    financial_trading_inputs ={
    'stock_selection': 'AAPL',
    'initial_capital': '100000',
    'risk_tolerance': 'Medium',
    'trading_strategy_preference': 'Day Trading',
    'news_impact_consideration': True
    }
    print(run_investment_crew(input_data=financial_trading_inputs))