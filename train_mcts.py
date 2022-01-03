from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

import combined_policy.mcts_policy as mcts
import combined_policy.ppo_policy as ppo
import combined_policy.common_policy as cp
from env import AbfahrtEnv
import utils
from tqdm import tqdm
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_mcts(env, mcts_trainer, ppo_model):
    env.training = "mcts"
    env = mcts.MCTSWrapper(env)
    observation = env.reset()
    snapshot = env.get_snapshot()
    root = mcts.Root(snapshot, observation)
    mcts_trainer.learn(root)


def train_ppo(env, mcts_model, ppo_model):
    # env.set_training(mcts_model=mcts_model)
    env.training = "ppo"
    ppo_model.learn(config.total_steps, reset_num_timesteps=False)


def eval_both(env, mcts_model, ppo_model):
    pass


n_episodes = 20
n_episodes_per_eval = 10
batch_size = 10
n_steps = 20
if __name__ == '__main__':
    config = utils.ConfigParams()
    train_env = AbfahrtEnv(config=config, mode="train", using_mcts=True)
    train_env.reset()
    # train_env = mcts.MCTSWrapper(train_env)
    # train_env = make_vec_env(lambda: train_env, n_envs=config.n_envs)
    # train_env = VecNormalize(train_env, norm_obs=False, gamma=config.gamma)

    eval_env = AbfahrtEnv(config=config, mode="eval", using_mcts=True)
    eval_env.reset()
    # eval_env = make_vec_env(lambda: eval_env, n_envs=1)
    # eval_env = VecNormalize(eval_env, norm_obs=False, gamma=config.gamma, training=False)

    value_net = cp.ValueNet(config).to(device)
    pi_net = mcts.PolicyNet(config).to(device)
    ppo_model = ppo.get_model(train_env, config=config, value_net=value_net)
    mcts_trainer = mcts.Trainer(train_env, value_net, pi_net, ppo_model.predict, config)

    train_env.get_ppo_action = ppo_model.predict
    train_env.get_mcts_action = mcts_trainer.predict
    eval_env.get_ppo_action = ppo_model.predict
    eval_env.get_mcts_action = mcts_trainer.predict
    
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    train_mcts(train_env, mcts_trainer, ppo_model)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()

    # for i in tqdm(range(n_episodes)):
    #     if i % 2 == 0:
    #         train_mcts(train_env, mcts_trainer, ppo_model)
    #     else:
    #         train_ppo(train_env, mcts_trainer, ppo_model)

    #     if i % n_episodes_per_eval == 0:
    #         eval_both(train_env, mcts_trainer, ppo_model)
