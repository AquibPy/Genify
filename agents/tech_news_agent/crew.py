from crewai import Crew,Process
from .tasks import research_task,write_task
from .agents import news_researcher,news_writer

crew=Crew(
    agents=[news_researcher,news_writer],
    tasks=[research_task,write_task],
    process=Process.sequential,

)

def run_crew(topic):
    result = crew.kickoff(inputs={'topic': topic})
    return str(result)

if __name__=='__main__':
    print(run_crew(topic="AI in Constructions"))