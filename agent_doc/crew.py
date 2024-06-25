from crewai import Crew
from .agents import treatment_advisor, diagnostician
from .tasks import diagnose_task, treatment_task


doc_crew = Crew(
    agents=[diagnostician, treatment_advisor],
    tasks=[diagnose_task, treatment_task],
    verbose=2
)

def run_doc_crew(input_data):
    result = doc_crew.kickoff(inputs=input_data)
    return result

if __name__=='__main__':
    doc_agent_input ={
    'gender': 'Male',
    'age': '28',
    'symptoms': 'fever, cough, headache',
    'medical_history': 'diabetes, hypertension'
    }
    print(run_doc_crew(input_data=doc_agent_input))