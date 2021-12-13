from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

import callbacks
import utils
from enviroments.env import AbfahrtEnv
import wandb


def train():
    def _train():
        config = utils.ConfigParams(wandb.config if USE_WANDB else None)

        env = AbfahrtEnv(config=config)
        env.reset()
        multi_env = make_vec_env(lambda: env, n_envs=N_ENVS)
        # eval_envs = make_vec_env(lambda: env, n_envs=1)
        eval_envs = env

        model = config.policy.get_model(multi_env, config, BATCH_SIZE, N_STEPS)
        logger = configure()
        logger.level = 10
        model.set_logger(logger=logger)
        callback = callbacks.get_callbacks(envs=eval_envs, logger=logger, use_wandb=USE_WANDB, n_steps=N_STEPS)
        model.learn(TOTAL_STEPS, callback=callback)

    if USE_WANDB:
        with wandb.init(save_code=False) as run:
            try: _train()
            except Exception as e: print(e); raise e
    else:
        _train()


VERBOSE = 1
USE_WANDB = 0

N_ENVS = 4
BATCH_SIZE = 16
N_STEPS = 256
TOTAL_STEPS = BATCH_SIZE * N_STEPS * 24

if __name__ == "__main__":
    if USE_WANDB:
        sweep_id = "schmalegg/schmalegger-hbf/qt2uzv7z"
        wandb.agent(sweep_id, function=train)
    else:
        train()
