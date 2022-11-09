# start by pulling the python image
FROM python:3.7

RUN pip install --upgrade pip

RUN apt-get update \
    && apt-get -y install tesseract-ocr 

WORKDIR /app

ENV FLASK_APP=app.py 

ENV FLASK_ENV=development

COPY ./requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 5000

COPY . .

CMD [ "python", "app.py" ]




