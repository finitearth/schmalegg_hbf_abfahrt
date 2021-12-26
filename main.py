from objects import EnvBlueprint
import env


fp = "input/input5.txt"
if __name__ == '__main__':
    env_bp = EnvBlueprint()
    env_bp.read_txt(fp)
    env_bp.render()

    # env_inf = env.AbfahrtEnv(mode="inference", config=model.config)
    # env_inf.inference_env = env_bp
