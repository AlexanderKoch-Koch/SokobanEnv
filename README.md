# Fast Sokoban Environment for Deep Reinforcement Learning
Sokoban is a simple game which requires a lot of planning. Since is doesn't require any fancy graphics it can be implemented very fast. One step in the SokobanENv takes about 10 microseconds. The game levels are created from a text file. Deepmind has published about 1,000,000 of these puzzles (https://github.com/deepmind/boxoban-levels).

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