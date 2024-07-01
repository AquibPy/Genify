from crewai import Crew
from .tasks import MLTask
from .agents import MLAgents
import pandas as pd

def run_ml_crew(file_path, user_question, model="llama3-70b-8192"):
    try:
        df = pd.read_csv(file_path).head(5)
    except Exception as e:
        return {"error": f"Error reading the file: {e}"}

    # Initialize agents and tasks
    tasks = MLTask()
    agents = MLAgents(model=model)

    problem_definition_agent = agents.problem_definition_agent()
    data_assessment_agent = agents.data_assessment_agent()
    model_recommendation_agent = agents.model_recommendation_agent()
    starter_code_agent = agents.starter_code_agent()

    task_define_problem = tasks.task_define_problem(problem_definition_agent)
    task_assess_data = tasks.task_assess_data(data_assessment_agent)
    task_recommend_model = tasks.task_recommend_model(model_recommendation_agent)
    task_generate_code = tasks.task_generate_code(starter_code_agent)

    # Format the input data for agents
    input_data = {
        "ml_problem": user_question,
        "df": df.head(),
        "file_name": file_path
    }

    # Initialize and run the crew
    ml_crew = Crew(
        agents=[problem_definition_agent, data_assessment_agent, model_recommendation_agent, starter_code_agent],
        tasks=[task_define_problem, task_assess_data, task_recommend_model, task_generate_code],
        verbose=True
    )

    result = ml_crew.kickoff(input_data)
    return result

if __name__=="__main__":
    print(run_ml_crew(file_path="data/iris.csv",
                       user_question="""
                       I have the iris dataset and I would like to build a machine learning model to classify the species of iris flowers based on their sepal and petal measurements.
                       The dataset contains four features: sepal length, sepal width, petal length, and petal width.
                       The target variable is the species of the iris flower,which can be one of three types: Setosa, Versicolor, or Virginica.
                       I would like to know the most suitable model for this classification problem and also get some starter code for the project.
                       """,
                       model="mixtral-8x7b-32768"))

