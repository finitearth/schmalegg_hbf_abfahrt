import sys
import traceback

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import callbacks
import utils
from env import AbfahrtEnv
import wandb


def train():
    def _train():
        config = utils.ConfigParams(wandb.config if USE_WANDB else None)

        train_env = AbfahrtEnv(config=config, mode="train")
        train_env.reset()
        train_env = make_vec_env(lambda: train_env, n_envs=config.n_envs)
        train_env = VecNormalize(train_env, norm_obs=False, gamma=config.gamma)

        eval_envs = AbfahrtEnv(config=config, mode="eval")
        eval_envs.reset()
        eval_envs = make_vec_env(lambda: eval_envs, n_envs=1)
        eval_envs = VecNormalize(eval_envs, norm_obs=False, gamma=config.gamma, training=False)

        model = config.policy.get_model(train_env, config)
        logger = utils.CustomLogger(USE_WANDB)
        logger.level = 10
        model.set_logger(logger=logger)
        callback = callbacks.get_callbacks(envs=eval_envs, use_wandb=USE_WANDB, config=config)
        model.learn(config.total_steps, callback=callback, eval_freq=10)
        model.save("models/v0")

    if USE_WANDB:
        with wandb.init(save_code=False) as run:
            try:
                _train()
            except Exception as e:
                ex_type, ex_value, ex_traceback = sys.exc_info()
                trace_back = traceback.extract_tb(ex_traceback)

                stack_trace = list()

                for trace in trace_back:
                    stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (
                        trace[0], trace[1], trace[2], trace[3]))
                print(ex_type.__name__)
                for st in stack_trace:
                    print(st)
                raise e


    else:
        _train()




VERBOSE = 1
USE_WANDB = 0

if __name__ == "__main__":
    if USE_WANDB:
        sweep_id = "wandb agent schmalegg/schmalegger-hbf/w1lzhdhl"
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
