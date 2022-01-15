FROM python:3.8-slim

# Install pip requirements
ADD . .
COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch_geometric
#RUN pip3 install torch_scatter

WORKDIR /app
COPY . /app

CMD ["python", "main.py"]
