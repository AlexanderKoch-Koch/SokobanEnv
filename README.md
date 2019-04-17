# Fast Sokoban Environment for Deep Reinforcement Learning

## Installation
```
git clone https://github.com/AlexanderKoch-Koch/SokobanEnv.git
cd SokobanEnv
pip install .
```

## Usage
import and initilization
```
import sokoban_env
env = sokoban_env.SokobanEnv()
```

start next level
```
observation = env.reset()
```
take an action (0 -> move right; 1 -> move down; 2 -> move left; 3 -> move up)
```
new_observation, reward, is_done, info = env.step(action)
```
