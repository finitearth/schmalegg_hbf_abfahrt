FROM python:3.8

ADD train_sweep.py .
<<<<<<< Updated upstream
ADD requirements.txt / 
ADD callbacks.py /
ADD utils.py / 
ADD enviroments /
ADD enviroments/env.py enviroments
ADD enviroments/env2.py enviroments
ADD enviroments/env3.py enviroments
ADD enviroments/env4.py enviroments
ADD policies /
ADD main.py / 
ADD objects.py / 
ADD Pipfile / 
ADD Pipfile.lock /
=======
ADD requirements.txt .
>>>>>>> Stashed changes



RUN pip install numpy
RUN pip install matplotlib
RUN pip install torch
RUN pip install stable_baselines3
RUN pip install torch_geometric
RUN pip install gym
RUN pip install wandb
RUN pip install torch_scatter
RUN pip install networkx
RUN pip install torch_sparse 


CMD ["python", "./train_sweep.py"]

