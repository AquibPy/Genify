import sqlite3
import random
import string

class EmployeeDatabaseManager:
    def __init__(self, db_name='employees.db'):
        self.db_name = db_name
        self.connection = None
        self.cursor = None

    def connect(self):
        self.connection = sqlite3.connect(self.db_name)
        self.cursor = self.connection.cursor()

    def close(self):
        if self.connection:
            self.connection.close()

    def create_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS employees (
                                Employee_ID INTEGER PRIMARY KEY,
                                Name TEXT,
                                Department TEXT,
                                Title TEXT,
                                Email TEXT,
                                City TEXT,
                                Salary INTEGER,
                                Work_Experience INTEGER
                            )''')

    def random_string(self, length):
        letters = string.ascii_letters
        return ''.join(random.choice(letters) for _ in range(length))

    def random_email(self):
        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'example.com']
        return f"{self.random_string(6)}_{self.random_string(6)}@{random.choice(domains)}"

    def generate_employee(self, used_employee_ids):
        first_names = ['John', 'Emma', 'Michael', 'Sophia', 'Matthew', 'Olivia']
        last_names = ['Smith', 'Johnson', 'Brown', 'Taylor', 'Williams', 'Jones']
        departments = ['HR', 'Finance', 'IT', 'Marketing', 'Operations']
        job_titles = {
            'HR': ['HR Manager', 'HR Coordinator', 'HR Assistant'],
            'Finance': ['Financial Analyst', 'Finance Manager', 'Accountant'],
            'IT': ['Software Engineer', 'IT Specialist', 'System Administrator'],
            'Marketing': ['Marketing Manager', 'Marketing Coordinator', 'Digital Marketer'],
            'Operations': ['Operations Manager', 'Operations Coordinator', 'Operations Analyst']
        }

        employee_id = random.randint(10000, 99999)
        while employee_id in used_employee_ids:  # Regenerate employee ID if duplicate
            employee_id = random.randint(10000, 99999)
        used_employee_ids.add(employee_id)
        name = random.choice(first_names) + " " + random.choice(last_names)
        department = random.choice(departments)
        title = random.choice(job_titles[department])
        email = self.random_email()
        city = random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])
        salary = random.randint(30000, 150000)  # Random salary between $30,000 and $150,000
        work_experience = random.randint(0, 20)  # Random work experience between 0 and 20 years
        return (employee_id, name, department, title, email, city, salary, work_experience)

    def insert_data(self):
        used_employee_ids = set()
        for _ in range(100):
            employee_data = self.generate_employee(used_employee_ids)
            self.cursor.execute('''INSERT INTO employees (Employee_ID, Name, Department, Title, Email, City, Salary, Work_Experience)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', employee_data)
        self.connection.commit()
        print("Data inserted successfully!")

    def create_database(self):
        self.connect()
        self.create_table()
        self.insert_data()
        self.close()


if __name__=='__main__':
    # Usage example:
    db_manager = EmployeeDatabaseManager()
    db_manager.create_database()
