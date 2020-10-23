from sokoban_env.sokoban_env import SokobanEnv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

env = SokobanEnv("../boxoban-levels/medium/train/", tiny_observation=False)


observation = env.reset()

done = False

while not done:
    action = int(input())
    observation, reward, done, _ = env.step(action)
    plt.imshow(observation)
    plt.show()
    print(action)
    print(reward)