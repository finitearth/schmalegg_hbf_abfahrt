from objects import EnvBlueprint

if __name__ == '__main__':
    env = EnvBlueprint()
    env.read_txt("input/input5.txt")
    env.render()
