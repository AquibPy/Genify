from crewai import Task

class MLTask():
    def task_define_problem(self,agent):
        return Task(
            description="""Clarify and define the machine learning problem, 
            including identifying the problem type and specific requirements.
               
            Here is the user's problem:
           {ml_problem}
           """,
           agent=agent,
           expected_output="A clear and concise definition of the machine learning problem."
        )

    def task_assess_data(self,agent):
        return Task(
            description="""Evaluate the user's data for quality and suitability, 
            suggesting preprocessing or augmentation steps if needed.

            Here is a sample of the user's data:
            {df}

            """,
            agent=agent,
            expected_output="An assessment of the data's quality and suitability, with suggestions for preprocessing or augmentation if necessary."
            )

    def task_recommend_model(self,agent):
        return Task(
            description="""Suggest suitable machine learning models for the defined problem 
            and assessed data, providing rationale for each suggestion.""",
            agent=agent,
            expected_output="A list of suitable machine learning models for the defined problem and assessed data, along with the rationale for each suggestion."
        )

    def task_generate_code(self,agent):
        return Task(
            description="""Generate starter Python code tailored to the user's project using the model recommendation agent's recommendation(s), 
            including snippets for package import, data handling, model definition, and training. """,
            agent=agent,
            expected_output="Python code snippets for package import, data handling, model definition, and training, tailored to the user's project, plus a brief summary of the problem and model recommendations."
        )