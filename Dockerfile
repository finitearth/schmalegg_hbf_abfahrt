# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim

EXPOSE 8000
ENV Var1=10

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
ADD read_input_files.py .
ADD env_tensor_for_real.py . 
ADD mcts/mcts.py . 

COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install torch_geometric
#RUN pip3 install torch_scatter

WORKDIR /app
COPY . /app
COPY /read_input_files.py /
ENTRYPOINT [ "/read_input_files.py" ]


CMD ["python", "env_tensor.py"]

EXPOSE 8000
