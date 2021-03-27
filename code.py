import pyvirtualdisplay
_display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
_ = _display.start()
import gym
from gym.utils.play import play
env = gym.make("CartPole-v0")
play(env, zoom=4)
import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer

ray.shutdown()
ray.init(
    include_webui=False,
    ignore_reinit_error=True,
    object_store_memory=8 * 1024 * 1024 * 1024  # 8GB limit … feel free to increase this if you can
)

ENV = 'Humanoid-v1'
TARGET_REWARD = 195
TRAINER = DQNTrainer

tune.run(
     TRAINER,
     stop={"episode_reward_mean": TARGET_REWARD},  # stop as soon as we "solve" the environment
     config={
       "env": ENV,
       "num_workers": 0,  # run in a single process
       "num_gpus": 0,
       "monitor": True,  # store stats and videos periodically
       "evaluation_num_episodes": 25,  # every 25 episodes instead of the default 10
     }
)
from base64 import b64encode
from pathlib import Path
from typing import List

# this will depend on which provider you are using; the correct version is
# probably what you get if you append /ray/results/ to the output from !pwd
OUT_PATH = Path('/root/ray_results/')

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
  return f'<p>{videopath.name}</p><video width=400 controls><source src="data:video/mp4;base64,{base64_encoded_mp4}" type="video/mp4"></video>'


from IPython.display import HTML
html = render_mp4(latest_videos()[-1])
HTML(html)