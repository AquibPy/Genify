from fastapi.testclient import TestClient
from api import app
from mongo import MongoDB

client = TestClient(app)
db = MongoDB()


def test_invoice_extractor():
    image_file = open("data\invoice.png", "rb")
    files = {"image_file": image_file}
    data = {"prompt": "To whom this invoice belongs"}

    response = client.post("/invoice_extractor", files=files, data=data)
    assert response.status_code == 200
    assert "response" in response.json()

def test_blogs():
    data = {"topic": "Tensorflow"}

    response = client.post("/blog_generator", data=data)
    assert response.status_code == 200
    assert "response" in response.json()

def test_sql_query():
    data = {"prompt": "Tell me the employees living in city Noida"}

    response = client.post("/Text2SQL", data=data)
    assert response.status_code == 200
    assert "response" in response.json()

def test_youtube_video_transcribe_summarizer():
    data = {"url": "https://www.youtube.com/watch?v=voexVsTHPBY&ab_channel=NabeelNawab"}

    response = client.post("/youtube_video_transcribe_summarizer", data=data)
    assert response.status_code == 200
    assert "response" in response.json()

def test_nutritionist_expert():
    image_file = open(r"data\burger.jpg", "rb")
    data = {"height": "165", "weight": "70"}

    response = client.post("/nutritionist_expert", files={"image_file": image_file}, data=data)
    assert response.status_code == 200
    assert "response" in response.json()

def test_talk2PDF():
    pdf_file = open(r"data\yolo.pdf", "rb")
    data = {"prompt": "Summary in 200 words"}

    response = client.post("/talk2PDF", files={"pdf": pdf_file}, data=data)
    assert response.status_code == 200
    assert "response" in response.json()

def test_questions_generator():
    pdf_file = open(r"data\yolo.pdf", "rb")

    # Make a request to the questions_generator endpoint
    response = client.post("/questions_generator", files={"pdf": pdf_file})
    assert response.status_code == 200
    assert "response" in response.json()