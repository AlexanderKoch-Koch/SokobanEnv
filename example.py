from sokoban_env import SokobanEnv
import random
import matplotlib.pyplot as plt

env = SokobanEnv()


observation = env.reset()

done = False

while not done:
    action = int(input())
    observation, reward, done, _ = env.step(action)
    print(action)
    print(reward)
    plt.imshow(observation)
    plt.show()