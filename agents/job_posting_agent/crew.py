from crewai import Crew
from dotenv import load_dotenv
load_dotenv()
from .tasks import Tasks
from .agents import JobAgents

tasks = Tasks()
agents = JobAgents()

researcher_agent = agents.research_agent()
writer_agent = agents.writer_agent()
review_agent = agents.review_agent()

research_company_culture_task = tasks.research_company_culture_task(researcher_agent)
industry_analysis_task = tasks.industry_analysis_task(researcher_agent)
research_role_requirements_task = tasks.research_role_requirements_task(researcher_agent)
draft_job_posting_task = tasks.draft_job_posting_task(writer_agent)
review_and_edit_job_posting_task = tasks.review_and_edit_job_posting_task(review_agent)

job_crew = Crew(
    agents=[researcher_agent, writer_agent, review_agent],
    tasks=[
        research_company_culture_task,
        industry_analysis_task,
        research_role_requirements_task,
        draft_job_posting_task,
        review_and_edit_job_posting_task
    ],
    verbose=True
)


def run_job_crew(input_data):
    result = job_crew.kickoff(input_data)
    return str(result)

if __name__=='__main__':
    job_agent_input = {
    'company_description': 'Microsoft is a global technology company that develops, manufactures, licenses, supports, and sells a wide range of software products, services, and devices, including the Windows operating system, Office suite, Azure cloud services, and Surface devices.',
    'company_domain': 'https://www.microsoft.com/',
    'hiring_needs': 'Data Scientist',
    'specific_benefits': 'work from home, medical insurance, generous parental leave, on-site fitness centers, and stock purchase plan'
}

    print(run_job_crew(job_agent_input))