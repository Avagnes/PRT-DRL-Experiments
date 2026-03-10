from rl_zoo3.train import train
import sys

sys.argv = sys.argv[:1] + [
    '-f', 'models',
    '--algo', 'ppo',
    '--env', 'LunarLanderContinuous-v3',
    '--device', 'cpu',
    '-P'
]

train()