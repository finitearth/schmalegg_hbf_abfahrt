from enviroments.generate_envs import generate_random_env
from utils import create_nx_graph
from objects import EnvBlueprint
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = EnvBlueprint()
    env.read_txt("input/input2.txt")
    env.render()