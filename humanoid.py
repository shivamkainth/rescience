import pyvirtualdisplay


_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()

import gym
import ray
from ray import tune
from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env

ray.shutdown()
ray.init( ignore_reinit_error=True)


ENV = 'HumanoidBulletEnv-v0'
def make_env(env_config):
    import pybullet_envs
    env = gym.make(ENV)
    return env
register_env(ENV, make_env)


TARGET_REWARD = 6000
TRAINER = SACTrainer


tune.run(
    TRAINER,
    stop={"episode_reward_mean": TARGET_REWARD},
    config={
        "env": ENV,
        "num_workers": 11,
        "num_gpus": 0,
        "monitor": True,
        "evaluation_num_episodes": 50,
        "optimization": {
            "actor_learning_rate": 1e-3,
            "critic_learning_rate": 1e-3,
            "entropy_learning_rate": 3e-4,
        },
        "train_batch_size": 128,
        "target_network_update_freq": 1,
        "learning_starts": 1000,
        "buffer_size": 1_000_000,
        "observation_filter": "MeanStdFilter",
    }
)
