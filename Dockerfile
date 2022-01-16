FROM python:3.8-slim

# Install pip requirements
ADD . .
COPY requirements.txt .

ARG input_file

RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch_geometric
RUN pip install sys



WORKDIR /app
COPY . /app
COPY . . 

CMD ["python", "main.py", input_file]
