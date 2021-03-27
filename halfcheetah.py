
import pyvirtualdisplay

_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()
from gym import envs
import gym
import ray
import pybullet_envs
from ray import tune
from ray.rllib.agents.sac import SACTrainer
from ray.tune.registry import register_env


_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()

import ray
from ray import tune
from ray.rllib.agents.sac import SACTrainer

ray.shutdown()
ray.init(ignore_reinit_error=True)

ENV = 'HalfCheetahBulletEnv-v0'
import gym
from ray.tune.registry import register_env
def make_env(env_config):
    import pybullet_envs
    return gym.make(ENV)
register_env(ENV, make_env)
TARGET_REWARD = 6000
TRAINER = SACTrainer

tune.run(
    TRAINER,
    stop={"episode_reward_mean": TARGET_REWARD},
    config={
        "env": ENV,
        "num_workers": 7,
        "num_gpus": 0,
        "monitor": True,
        "evaluation_num_episodes": 50,
    }
)

from base64 import b64encode
from pathlib import Path
from typing import List

# this will depend on which provider you are using; the correct version is
# probably what you get if you append /ray/results/ to the output from !pwd
OUT_PATH = Path('/home/A')

def latest_experiment() -> Path:
  """ Get the path of the results directory of the most recent training run. """
  experiment_dirs = []
  for algorithm in OUT_PATH.iterdir():
    if not algorithm.is_dir():
      continue
    for experiment in algorithm.iterdir():
      if not experiment.is_dir():
        continue
      experiment_dirs.append((experiment.stat().st_mtime, experiment))
  return max(experiment_dirs)[1]

def latest_videos() -> List[Path]:
  # because the ISO timestamp is in the name, the last alphabetically is the latest
  return list(sorted(latest_experiment().glob('*.mp4')))

def render_mp4(videopath: Path) -> str:
  mp4 = open(videopath, 'rb').read()
  base64_encoded_mp4 = b64encode(mp4).decode()
  return f'<p>{videopath.name}</p><video width=400 controls>  <source src="data:video/mp4;base64,{base64_encoded_mp4}" type="video/mp4"></video>'
