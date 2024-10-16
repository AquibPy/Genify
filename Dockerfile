FROM python:3.9-slim


WORKDIR /app


COPY requirements.txt .

RUN apt-get update && apt-get install -y libmagic-dev

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt


COPY . .


EXPOSE 8000


ENV PORT=8000

# Command to run the application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]