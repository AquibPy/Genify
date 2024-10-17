FROM python:3.12.7-alpine

WORKDIR /app

COPY requirements.txt .

# RUN apt-get update && apt-get install -y libmagic-dev

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "10000"]