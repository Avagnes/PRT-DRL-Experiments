from rl_zoo3.train import train
import sys

sys.argv = sys.argv[:1] + [
    '-f', 'models',
    '--algo', 'tqc',
    '--env', 'MountainCarContinuous-v0',
    '--device', 'cuda',
    '-P'
]

train()