from sokoban_env import SokobanEnv
import random
import matplotlib.pyplot as plt

env = SokobanEnv("./train/")


observation = env.reset()

done = False

while not done:
    action = int(input())
    observation, reward, done, _ = env.step(action, tiny_observation=False)
    plt.imshow(observation)
    plt.show()
    print(action)
    print(reward)
    env.render()