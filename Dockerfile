FROM python:3.8

ADD train_sweep.py .
ADD requirements.txt . 

RUN pip install -r requirements.txt

CMD ["python", "./train_sweep.py", "./requirements.txt"]