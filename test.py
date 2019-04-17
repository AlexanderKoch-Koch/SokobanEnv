import gym
import gym_sokoban
from Models import simple_DNN
import tensorflow as tf
import numpy as np
from tensorboard_logger import configure, log_value
from Experience import Experience
import random

configure("runs/8")
# log_value('v1', v1, step)

episodes = 100000
eta = 0.1
gamma = 0.95
env = gym.make('Sokoban-v0')#mode='tiny_rgb_array')
q_net = simple_DNN(architecture=[300, 30, 20, 4])
q_net_target = simple_DNN(architecture=[300, 30, 20, 4])
memory = Experience(10000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for episode in range(episodes):
        if episode == 999:
            i = 1
        done = False
        observation = env.reset().flatten()
        step = 0
        reward_sum = 0
        while not done:
            if random.randint(0, 100) > (eta * 100):
                action = np.argmax(q_net.predict(sess, observation))
            else:
                action = random.randint(0, 3)
            observation_next, reward, done, info = env.step(action)
            observation_next = observation_next.flatten()
            memory.store(observation, action, reward, observation_next, done, 100)
            observation = observation_next
            step += 1
            # env.render()
            reward_sum += reward

        log_value("avg_reward", reward_sum/step, episode)

    for i in range(20):
        [observation, action, reward, observation_next, done, priority], index = memory.sample(prioritized_replay=True)
        q_state = q_net.predict(sess, observation)
        target = np.copy(q_state)

        if done:
            target[action] = reward
        else:
            q_next = q_net_target.predict(sess, observation_next)
            target[action] = reward + gamma * np.max(q_next)

        td_error = np.abs(q_state[action] - target[action])
        memory.update_td_error(index, td_error)
        loss = q_net.train(sess, observation, target)

    if episode % 20 == 0:
        # update weights of target network every n episodes
        q_net_target.assign_weights_from(sess, q_net)
