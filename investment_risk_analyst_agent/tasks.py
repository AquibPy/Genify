from crewai import Task
from investment_risk_analyst_agent.tools import search_tool,scrape_tool
from investment_risk_analyst_agent.agents import data_analyst_agent,trading_strategy_agent,execution_agent,risk_management_agent

data_analysis_task = Task(
    description=(
        "Continuously monitor and analyze market data for the selected stock ({stock_selection}). "
        "Employ statistical modeling and machine learning techniques to identify trends and predict market movements."
    ),
    expected_output=(
        "Generate insights and alerts regarding significant market opportunities or threats for {stock_selection}."
    ),
    agent=data_analyst_agent,
    tools=[scrape_tool, search_tool],
)

strategy_development_task = Task(
    description=(
        "Develop and refine trading strategies based on insights from the Data Analyst and user-defined risk tolerance ({risk_tolerance}). "
        "Incorporate trading preferences ({trading_strategy_preference}) in the strategy development process."
    ),
    expected_output=(
        "A set of potential trading strategies for {stock_selection} that align with the user's risk tolerance."
    ),
    agent=trading_strategy_agent,
    tools=[scrape_tool, search_tool],
)

execution_planning_task = Task(
    description=(
        "Analyze approved trading strategies to determine the optimal execution methods for {stock_selection}, "
        "considering current market conditions and pricing strategies."
    ),
    expected_output=(
        "Comprehensive execution plans detailing how and when to execute trades for {stock_selection}."
    ),
    agent=execution_agent,
)

risk_assessment_task = Task(
    description=(
        "Evaluate the risks associated with the proposed trading strategies and execution plans for {stock_selection}. "
        "Provide a detailed analysis of potential risks and recommend mitigation strategies."
    ),
    expected_output=(
        "A comprehensive risk analysis report detailing potential risks and mitigation recommendations for {stock_selection}."
    ),
    agent=risk_management_agent,
    tools=[scrape_tool, search_tool],
)
