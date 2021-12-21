from stable_baselines3.common.env_util import make_vec_env

import callbacks
import utils
# from enviroments.env import AbfahrtEnv
from enviroments.env_from_files import AbfahrtEnv
import wandb


def train():
    def _train():
        config = utils.ConfigParams(wandb.config if USE_WANDB else None)

        train_env = AbfahrtEnv(config=config, mode="eval")
        train_env.mode = "train"
        train_env.reset()
        multi_env = make_vec_env(lambda: train_env, n_envs=config.n_envs)
        eval_envs = AbfahrtEnv(config=config)
        eval_envs.mode = "eval"
        eval_envs.reset()
        eval_envs = make_vec_env(lambda: eval_envs, n_envs=8)

        model = config.policy.get_model(multi_env, config)
        logger = utils.CustomLogger(USE_WANDB)
        logger.level = 10
        model.set_logger(logger=logger)
        callback = callbacks.get_callbacks(envs=eval_envs, use_wandb=USE_WANDB, config=config)
        model.learn(config.total_steps, callback=callback)

    if USE_WANDB:
        with wandb.init(save_code=False) as run:
            try: _train()
            except Exception as e: print(str(e)); raise e
    else:
        _train()


VERBOSE = 1
USE_WANDB = 0

if __name__ == "__main__":
    if USE_WANDB:
        sweep_id = "wandb agent schmalegg/schmalegger-hbf/hcglzsc3"
        sweep_id = sweep_id.split("agent ")[1]
        wandb.agent(sweep_id, function=train)
    else:
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        train()
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('tottime')
        stats.print_stats()
